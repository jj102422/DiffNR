import os
import json
import gc
import sys
import lpips
import clip
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

import wandb
from cleanfid.fid import get_folder_features, build_feature_extractor, fid_from_feats
# from pix2pix_turbo import Pix2Pix_Turbo
sys.path.append("./slicefixer") # 确保能索引到 DiffNR 里的 slicefixer 模块
from SliceFixer import SliceFixer
from my_utils.training_utils import parse_args_paired_training

try:
    sys.path.append("./r2_gaussian/submodules/fused-ssim")
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False


def normalize_to_255(img):
    # CT 切片不是天然落在 [-1, 1]；这里改成鲁棒的分位数窗宽/窗位归一化，
    # 避免把 0 值背景错误映射成中灰，同时也能压制极端值的影响。
    # 特别处理：对于有大量零值背景（如 GT volume）的情况，只对非零值进行分位数统计
    img = img.detach().cpu().float()
    if img.numel() == 0:
        return torch.zeros_like(img, dtype=torch.uint8)
    
    # 分离零值和非零值
    non_zero_mask = img != 0
    if non_zero_mask.sum() > 0:
        # 如果有非零值，对非零值进行分位数统计
        non_zero_vals = img[non_zero_mask]
        lo = torch.quantile(non_zero_vals, 0.01)
        hi = torch.quantile(non_zero_vals, 0.99)
    else:
        # 如果全是零，直接返回黑图
        return torch.zeros_like(img, dtype=torch.uint8)
    
    if torch.isclose(hi, lo):
        lo = non_zero_vals.min()
        hi = non_zero_vals.max()
    if torch.isclose(hi, lo):
        return torch.zeros_like(img, dtype=torch.uint8)
    
    img = img.clamp(lo, hi)
    img = (img - lo) / (hi - lo + 1e-8) * 255.0
    return img.clamp(0, 255).to(torch.uint8)


def unique_parameters(params):
    # 收集优化器参数时按对象 id 去重，避免同一个参数被重复加入参数组。
    unique = []
    seen = set()
    for param in params:
        param_id = id(param)
        if param_id in seen:
            continue
        seen.add(param_id)
        unique.append(param)
    return unique


class MedicalCTDataset(torch.utils.data.Dataset):
    def __init__(self, case_paths, tokenizer, prompt):
        self.case_paths = case_paths
        self.caption = prompt
        self.input_ids = tokenizer(
            prompt,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        self.index_map = []
        self._vol_cache = {}

        for case_path in self.case_paths:
            coarse_path = os.path.join(case_path, "vol_pred.npy")
            gt_candidates = [
                os.path.join(case_path, "volume_gt.npy"),
                os.path.join(case_path, "vol_gt.npy"),
            ]
            gt_path = next((p for p in gt_candidates if os.path.exists(p)), None)
            if not os.path.exists(coarse_path) or gt_path is None:
                continue
            coarse_vol = np.load(coarse_path, mmap_mode="r")
            gt_vol = np.load(gt_path, mmap_mode="r")
            if coarse_vol.ndim != 3 or gt_vol.ndim != 3:
                continue
            if coarse_vol.shape != gt_vol.shape:
                continue
            self._vol_cache[case_path] = (coarse_vol, gt_vol)
            # 这里的 volume_gt.npy / vol_pred.npy 已经在数据生成阶段转成 XYZ 顺序，
            # 因此 axis 2 才是 axial 方向；索引范围也要跟着改成 shape[2]。
            for slice_idx in range(coarse_vol.shape[2]):
                self.index_map.append((case_path, slice_idx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        case_path, slice_idx = self.index_map[idx]
        coarse_vol, gt_vol = self._vol_cache[case_path]
        # axial slice：axis 2 对应 Z 方向，因此这里沿最后一个维度取切片。
        coarse_slice = coarse_vol[:, :, slice_idx]
        gt_slice = gt_vol[:, :, slice_idx]

        # 这里需要 copy 一份，避免 mmap 读出的只读 numpy 数组直接转 tensor 时触发警告。
        coarse_tensor = torch.from_numpy(coarse_slice.copy()).float().unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_slice.copy()).float().unsqueeze(0)

        coarse_tensor = coarse_tensor.repeat(3, 1, 1)
        gt_tensor = gt_tensor.repeat(3, 1, 1)

        return {
            "conditioning_pixel_values": coarse_tensor,
            "output_pixel_values": gt_tensor,
            "caption": self.caption,
            "input_ids": self.input_ids,
        }


def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    net_pix2pix = SliceFixer(
        pretrained_name=None,
        pretrained_path=None,
        lora_rank_unet=args.lora_rank_unet,
        lora_rank_vae=args.lora_rank_vae,
        sd_turbo_path=args.pretrained_model_name_or_path,
    )
    net_pix2pix.set_train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_pix2pix.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_pix2pix.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gan_disc_type == "vagan_clip":
        import vision_aided_loss
        net_disc = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
    else:
        raise NotImplementedError(f"Discriminator type {args.gan_disc_type} not implemented")

    net_disc = net_disc.cuda()
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_clip, _ = clip.load("ViT-B/32", device="cuda")
    net_clip.requires_grad_(False)
    net_clip.eval()

    net_lpips.requires_grad_(False)

    # 只把需要训练的 LoRA / skip / conv 参数放进优化器，随后再做一次去重。
    layers_to_opt = []
    for n, _p in net_pix2pix.unet.named_parameters():
        if "lora" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt += list(net_pix2pix.unet.conv_in.parameters())
    for n, _p in net_pix2pix.vae.named_parameters():
        if "lora" in n and "vae_skip" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt = layers_to_opt + list(net_pix2pix.vae.decoder.skip_conv_1.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_2.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_3.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_4.parameters())
    # 防止 conv_in、skip_conv 或 LoRA 参数在不同收集路径中被重复加入。
    layers_to_opt = unique_parameters(layers_to_opt)

    # 【FP16 梯度修复】确保所有可训练参数保持 FP32，避免 GradScaler unscale 崩溃
    for name, param in net_pix2pix.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)

    optimizer_disc = torch.optim.AdamW(net_disc.parameters(), lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles, power=args.lr_power)

    info_path = args.info_json
    if not os.path.isabs(info_path):
        info_path = os.path.join(os.getcwd(), info_path)
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Missing info.json: {info_path}")

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    train_ids = info.get(args.train_split, [])
    val_ids = info.get(args.val_split, [])
    if not val_ids:
        val_ids = info.get("test", [])

    def to_case_paths(case_ids):
        return [os.path.join(args.dataset_folder, cid) for cid in case_ids]

    train_case_paths = to_case_paths(train_ids)
    val_case_paths = to_case_paths(val_ids)

    prompt = "high quality medical CT slice, clear anatomical structures"
    dataset_train = MedicalCTDataset(
        case_paths=train_case_paths,
        tokenizer=net_pix2pix.tokenizer,
        prompt=prompt,
    )
    dl_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    dataset_val = MedicalCTDataset(
        case_paths=val_case_paths,
        tokenizer=net_pix2pix.tokenizer,
        prompt=prompt,
    )
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # Prepare everything with our `accelerator`.
    net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc = accelerator.prepare(
        net_pix2pix, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc
    )
    net_clip, net_lpips = accelerator.prepare(net_clip, net_lpips)
    # renorm with image net statistics
    t_clip_renorm = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move all networks to weight_dtype (only change non-trainable params for net_pix2pix and net_disc)
    # 【FP16 梯度修复】由于 Accelerate 的 unscale 操作在 fp16 混合精度下会崩溃，
    # 我们仅将不参与训练的参数(requires_grad=False)转为 fp16 节省显存，
    # 强制让所有参与训练的参数(requires_grad=True)保持为 float32。
    net_lpips.to(dtype=weight_dtype)
    net_clip.to(dtype=weight_dtype)
    
    for name, param in net_pix2pix.named_parameters():
        if not param.requires_grad:
            param.data = param.data.to(weight_dtype)
            
    for name, param in net_disc.named_parameters():
        if not param.requires_grad:
            param.data = param.data.to(weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        init_kwargs = {"wandb": {"name": args.tracker_run_name}} if args.tracker_run_name else {}
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config, init_kwargs=init_kwargs)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
        disable=not accelerator.is_local_main_process,)

    # turn off eff. attn for the discriminator
    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False

    # compute the reference stats for FID tracking
    if accelerator.is_main_process and args.track_val_fid:
        feat_model = build_feature_extractor("clean", "cuda", use_dataparallel=False)

        def fn_transform(x):
            x_pil = Image.fromarray(x)
            out_pil = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.LANCZOS)(x_pil)
            return np.array(out_pil)

        ref_stats = get_folder_features(os.path.join(args.dataset_folder, "test_B"), model=feat_model, num_workers=0, num=None,
                shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                mode="clean", custom_image_tranform=fn_transform, description="", verbose=True)

    # start the training loop
    global_step = 0
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            l_acc = [net_pix2pix, net_disc]
            with accelerator.accumulate(*l_acc):
                x_src = batch["conditioning_pixel_values"].cuda()
                x_tgt = batch["output_pixel_values"].cuda()
                B, C, H, W = x_src.shape
                # forward pass
                x_tgt_pred = net_pix2pix(x_src, prompt_tokens=batch["input_ids"], deterministic=True)
                # Reconstruction loss
                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * args.lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * args.lambda_lpips

                # CLIP alignment loss
                loss_clipsim = torch.tensor(0.0, device=x_tgt_pred.device)
                if args.lambda_clipsim > 0:
                    x_tgt_pred_renorm = t_clip_renorm(x_tgt_pred * 0.5 + 0.5)
                    x_tgt_pred_renorm = F.interpolate(
                        x_tgt_pred_renorm,
                        (224, 224),
                        mode="bilinear",
                        align_corners=False,
                    )
                    caption_tokens = clip.tokenize(batch["caption"], truncate=True).to(x_tgt_pred.device)
                    clipsim, _ = net_clip(x_tgt_pred_renorm, caption_tokens)
                    loss_clipsim = (1 - clipsim.mean() / 100) * args.lambda_clipsim

                # SSIM loss
                if args.lambda_ssim > 0 and not FUSED_SSIM_AVAILABLE:
                    raise ValueError(
                        "fused_ssim is required for SSIM loss. Please install fused_ssim."
                    )
                loss_ssim = torch.tensor(0.0, device=x_tgt_pred.device)
                if args.lambda_ssim > 0:
                    pred_01 = (x_tgt_pred + 1.0) * 0.5
                    tgt_01 = (x_tgt + 1.0) * 0.5
                    ssim_val = fused_ssim(pred_01, tgt_01)
                    loss_ssim = (1.0 - ssim_val) * args.lambda_ssim

                # GAN loss for generator
                loss_gan = net_disc(x_tgt_pred, for_G=True).mean() * args.lambda_gan

                # Total generator loss (paper formula)
                loss = loss_l2 + loss_lpips + loss_clipsim + loss_gan + loss_ssim

                accelerator.backward(loss, retain_graph=False)
                
                # [修复 NaN 与 Unscale 报错]
                # 1. 重新启用梯度裁剪 (clip_grad_norm_)，避免模型在训练扩散和 GAN 时梯度爆炸产生 NaN 问题。
                # 2. 移除了之前错误屏蔽 fp16 裁剪的逻辑。
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                """
                Discriminator loss: fake image vs real image
                """
                # [修复 NaN 与 Unscale 报错]
                # 这是修复最核心死机Bug的地方。
                # 原逻辑分别对 real 和 fake 调用了 backward() 并执行了两次 step()，
                # 导致混合精度的 GradScaler 状态异常并直接报错或产生 NaN。
                # 正确的做法：真实和生成的 loss 相加，一个 batch 内只执行一次统一的 backward 和 step。
                # real vs fake
                lossD_real = net_disc(x_tgt.detach(), for_real=True).mean() * args.lambda_gan
                lossD_fake = net_disc(x_tgt_pred.detach(), for_real=False).mean() * args.lambda_gan
                
                lossD = lossD_real + lossD_fake
                accelerator.backward(lossD)

                # [修复 NaN 错误] 同样重新启用判别器的梯度裁剪，控制判别器的更新幅度
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {}
                    # log all the losses
                    logs["lossG"] = loss.detach().item()
                    logs["lossD"] = lossD.detach().item()
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    if args.lambda_clipsim > 0:
                        logs["loss_clipsim"] = loss_clipsim.detach().item()
                    if args.lambda_ssim > 0:
                        logs["loss_ssim"] = loss_ssim.detach().item()
                    if args.lambda_gan > 0:
                        logs["loss_gan"] = loss_gan.detach().item()
                    progress_bar.set_postfix(**logs)

                    # viz some images
                    if global_step % args.viz_freq == 1:
                        # 训练阶段把当前 batch 的 input / target / output 三个子图发到 wandb。
                        log_dict = {
                            "train/input": [wandb.Image(normalize_to_255(x_src[idx]), caption=f"input_{idx}") for idx in range(B)],
                            "train/target": [wandb.Image(normalize_to_255(x_tgt[idx]), caption=f"target(GT)_{idx}") for idx in range(B)],
                            "train/output": [wandb.Image(normalize_to_255(x_tgt_pred[idx]), caption=f"output_{idx}") for idx in range(B)],
                        }
                        for k in log_dict:
                            logs[k] = log_dict[k]

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(net_pix2pix).save_model(outf)

                    # compute validation set FID, L2, LPIPS, CLIP-SIM
                    if global_step % args.eval_freq == 1:
                        l_l2, l_lpips, l_clipsim = [], [], []
                        if args.track_val_fid:
                            os.makedirs(os.path.join(args.output_dir, "eval", f"fid_{global_step}"), exist_ok=True)
                        for step, batch_val in enumerate(dl_val):
                            if step >= args.num_samples_eval:
                                break
                            x_src = batch_val["conditioning_pixel_values"].cuda()
                            x_tgt = batch_val["output_pixel_values"].cuda()
                            B, C, H, W = x_src.shape
                            assert B == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                # forward pass
                                x_tgt_pred = accelerator.unwrap_model(net_pix2pix)(x_src, prompt_tokens=batch_val["input_ids"].cuda(), deterministic=True)
                                # compute the reconstruction losses
                                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean")
                                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean()
                                # compute clip similarity loss
                                x_tgt_pred_renorm = t_clip_renorm(x_tgt_pred * 0.5 + 0.5)
                                x_tgt_pred_renorm = F.interpolate(x_tgt_pred_renorm, (224, 224), mode="bilinear", align_corners=False)
                                caption_tokens = clip.tokenize(batch_val["caption"], truncate=True).to(x_tgt_pred.device)
                                clipsim, _ = net_clip(x_tgt_pred_renorm, caption_tokens)
                                clipsim = clipsim.mean()

                                l_l2.append(loss_l2.item())
                                l_lpips.append(loss_lpips.item())
                                l_clipsim.append(clipsim.item())
                            # save output images to file for FID evaluation
                            if args.track_val_fid:
                                output_pil = transforms.ToPILImage()(x_tgt_pred[0].cpu() * 0.5 + 0.5)
                                outf = os.path.join(args.output_dir, "eval", f"fid_{global_step}", f"val_{step}.png")
                                output_pil.save(outf)
                        if args.track_val_fid:
                            curr_stats = get_folder_features(os.path.join(args.output_dir, "eval", f"fid_{global_step}"), model=feat_model, num_workers=0, num=None,
                                    shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                                    mode="clean", custom_image_tranform=fn_transform, description="", verbose=True)
                            fid_score = fid_from_feats(ref_stats, curr_stats)
                            logs["val/clean_fid"] = fid_score
                        logs["val/l2"] = np.mean(l_l2)
                        logs["val/lpips"] = np.mean(l_lpips)
                        logs["val/clipsim"] = np.mean(l_clipsim)
                        
                        # 验证阶段也同步记录三张图，方便直接对比 input / target / output。
                        # 这里保留最后一个 val batch 的结果作为可视化样本。
                        logs["val/input"] = [wandb.Image(normalize_to_255(x_src[0]), caption="val_input")]
                        logs["val/target"] = [wandb.Image(normalize_to_255(x_tgt[0]), caption="val_target")]
                        logs["val/output"] = [wandb.Image(normalize_to_255(x_tgt_pred[0]), caption="val_output")]
                        
                        gc.collect()
                        torch.cuda.empty_cache()
                    accelerator.log(logs, step=global_step)


if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)
