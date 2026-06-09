import os
import requests
import sys
import copy
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from peft import LoraConfig
p = "slicefixer/"
sys.path.append(p)
from model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd, CrossAttnFusionAdapter


INTENSITY_DOMAIN = {
    "stored_volume": "v_saved = clip(HU, 0, None) / 3000; may exceed 1",
    "model_input": "v = clip(v_saved, 0, 1); s = 2 * v - 1",
    "model_output_inverse": "v = (clip(s, -1, 1) + 1) / 2",
    "upper_clipping": True,
    "model_range": [-1.0, 1.0],
}
XRAY_CONDITIONING = {
    "encoder": "microsoft/rad-dino",
    "feature": "cls_embedding",
    "per_view_shape": [1, 768],
}


class TwinConv(torch.nn.Module):
    def __init__(self, convin_pretrained, convin_curr):
        super(TwinConv, self).__init__()
        self.conv_in_pretrained = copy.deepcopy(convin_pretrained)
        self.conv_in_curr = copy.deepcopy(convin_curr)
        self.r = None

    def forward(self, x):
        x1 = self.conv_in_pretrained(x).detach()
        x2 = self.conv_in_curr(x)
        return x1 * (1 - self.r) + x2 * (self.r)


class SliceFixer(torch.nn.Module):
    def __init__(
        self,
        pretrained_name=None,
        pretrained_path=None,
        ckpt_folder="checkpoints",
        lora_rank_unet=8,
        lora_rank_vae=4,
        sd_turbo_path=None,
        use_xray_conditioning=False,
        conditioning_in_channels=3,
    ):
        super().__init__()
        loaded_sd = torch.load(pretrained_path, map_location="cpu") if pretrained_path is not None else None
        if loaded_sd is not None and conditioning_in_channels is None:
            conditioning_in_channels = loaded_sd.get("conditioning_in_channels", 3)
        self.conditioning_in_channels = int(conditioning_in_channels or 3)
        self.use_xray_conditioning = use_xray_conditioning
        self.sd_turbo_path = (
            sd_turbo_path
            or os.environ.get("SD_TURBO_PATH")
            or "stabilityai/sd-turbo"
        )
        # local checkpoint directory -> local_files_only=True, otherwise allow hub download.
        local_files_only = os.path.isdir(self.sd_turbo_path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.sd_turbo_path,
            subfolder="tokenizer",
            local_files_only=local_files_only,
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.sd_turbo_path,
            subfolder="text_encoder",
            local_files_only=local_files_only,
        ).cuda()
        
        self.fusion_adapter = CrossAttnFusionAdapter(text_dim=1024, xray_dim=768, num_heads=8).cuda()
        self.input_adapter = self._make_input_adapter(self.conditioning_in_channels)
        self.sched = make_1step_sched(self.sd_turbo_path, local_files_only)
        
        vae = AutoencoderKL.from_pretrained(
            self.sd_turbo_path,
            subfolder="vae",
            local_files_only=local_files_only,
        )
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.ignore_skip = False
        unet = UNet2DConditionModel.from_pretrained(
            self.sd_turbo_path,
            subfolder="unet",
            local_files_only=local_files_only,
        )

        if pretrained_name == "edge_to_image":
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/edge_to_image_loras.pkl"
            os.makedirs(ckpt_folder, exist_ok=True)
            outf = os.path.join(ckpt_folder, "edge_to_image_loras.pkl")
            if not os.path.exists(outf):
                print(f"Downloading checkpoint to {outf}")
                response = requests.get(url, stream=True)
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                with open(outf, 'wb') as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("ERROR, something went wrong")
                print(f"Downloaded successfully to {outf}")
            p_ckpt = outf
            sd = torch.load(p_ckpt, map_location="cpu")
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_name == "sketch_to_image_stochastic":
            # download from url
            url = "https://www.cs.cmu.edu/~img2img-turbo/models/sketch_to_image_stochastic_lora.pkl"
            os.makedirs(ckpt_folder, exist_ok=True)
            outf = os.path.join(ckpt_folder, "sketch_to_image_stochastic_lora.pkl")
            if not os.path.exists(outf):
                print(f"Downloading checkpoint to {outf}")
                response = requests.get(url, stream=True)
                total_size_in_bytes = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                with open(outf, 'wb') as file:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("ERROR, something went wrong")
                print(f"Downloaded successfully to {outf}")
            p_ckpt = outf
            convin_pretrained = copy.deepcopy(unet.conv_in)
            unet.conv_in = TwinConv(convin_pretrained, unet.conv_in)
            sd = torch.load(p_ckpt, map_location="cpu")
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_path is not None:
            sd = loaded_sd
            self.use_xray_conditioning = sd.get("use_xray_conditioning", True)
            if self.use_xray_conditioning and "state_dict_fusion_adapter" not in sd:
                raise ValueError(
                    "SliceFixer checkpoint does not contain RAD-DINO fusion adapter weights. "
                    "Retrain with the high-HU/RAD-DINO conditioning pipeline."
                )
            if sd.get("intensity_domain") != INTENSITY_DOMAIN:
                raise ValueError("SliceFixer checkpoint intensity domain is incompatible with this pipeline.")
            if self.use_xray_conditioning and sd.get("xray_conditioning") != XRAY_CONDITIONING:
                raise ValueError("SliceFixer checkpoint X-ray conditioning metadata is incompatible.")
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            _sd_vae = vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            vae.load_state_dict(_sd_vae)
            unet.add_adapter(unet_lora_config)
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)
            if self.use_xray_conditioning:
                self.fusion_adapter.load_state_dict(sd["state_dict_fusion_adapter"])
            if "state_dict_input_adapter" in sd:
                checkpoint_channels = int(sd.get("conditioning_in_channels", 3))
                if checkpoint_channels != self.conditioning_in_channels:
                    raise ValueError(
                        f"Checkpoint was trained with {checkpoint_channels} conditioning channels, "
                        f"but this run requested {self.conditioning_in_channels}."
                    )
                self.input_adapter.load_state_dict(sd["state_dict_input_adapter"])
            elif self.conditioning_in_channels != 3:
                print(
                    f"Initializing {self.conditioning_in_channels}->3 input adapter from center slice; "
                    "new context/mask channels are ignored until this adapter is fine-tuned."
                )
            self.lora_rank_unet = sd["rank_unet"]
            self.lora_rank_vae = sd["rank_vae"]
            self.target_modules_vae = sd["vae_lora_target_modules"]
            self.target_modules_unet = sd["unet_lora_target_modules"]

        elif pretrained_name is None and pretrained_path is None:
            print("Initializing model with random weights")
            torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
            torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
            target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                "to_k", "to_q", "to_v", "to_out.0",
            ]
            vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
                target_modules=target_modules_vae)
            vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            target_modules_unet = [
                "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
                "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
            ]
            unet_lora_config = LoraConfig(r=lora_rank_unet, init_lora_weights="gaussian",
                target_modules=target_modules_unet
            )
            unet.add_adapter(unet_lora_config)
            self.lora_rank_unet = lora_rank_unet
            self.lora_rank_vae = lora_rank_vae
            self.target_modules_vae = target_modules_vae
            self.target_modules_unet = target_modules_unet

        # unet.enable_xformers_memory_efficient_attention()
        unet.to("cuda")
        vae.to("cuda")
        self.unet, self.vae = unet, vae
        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([999], device="cuda").long() # difix3D used 199
        self.text_encoder.requires_grad_(False)

    def _make_input_adapter(self, in_channels):
        if in_channels == 3:
            return torch.nn.Identity()
        adapter = torch.nn.Conv2d(in_channels, 3, kernel_size=1, bias=False).cuda()
        with torch.no_grad():
            adapter.weight.zero_()
            slice_channels = in_channels // 2 if in_channels > 1 and in_channels % 2 == 0 else in_channels
            center_channel = slice_channels // 2
            for out_channel in range(3):
                adapter.weight[out_channel, center_channel, 0, 0] = 1.0
        return adapter

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.fusion_adapter.eval()
        self.fusion_adapter.requires_grad_(False)
        self.input_adapter.eval()
        self.input_adapter.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)
        if self.use_xray_conditioning:
            self.fusion_adapter.train()
            for p in self.fusion_adapter.parameters():
                p.requires_grad = True
        else:
            self.fusion_adapter.eval()
            self.fusion_adapter.requires_grad_(False)
        self.input_adapter.train()
        self.input_adapter.requires_grad_(True)

    def forward(self, c_t, prompt=None, prompt_tokens=None, deterministic=True, r=1.0, noise_map=None, xray_feat1=None, xray_feat2=None):
        # either the prompt or the prompt_tokens should be provided
        assert (prompt is None) != (prompt_tokens is None), "Either prompt or prompt_tokens should be provided"

        if prompt is not None:
            # encode the text prompt
            caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]
        
        if self.use_xray_conditioning:
            if xray_feat1 is None or xray_feat2 is None:
                raise ValueError("This SliceFixer model requires two RAD-DINO X-ray conditioning features.")
            xray_feats = torch.cat([xray_feat1, xray_feat2], dim=1)
            xray_feats = xray_feats.to(caption_enc.device, dtype=caption_enc.dtype)
            caption_enc = self.fusion_adapter(caption_enc, xray_feats)
        if deterministic:
            if c_t.shape[1] != self.conditioning_in_channels:
                raise ValueError(
                    f"SliceFixer expected {self.conditioning_in_channels} conditioning channels, "
                    f"got {c_t.shape[1]}."
                )
            c_t = self.input_adapter(c_t)
            encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
            model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc,).sample
            x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
            x_denoised = x_denoised.to(model_pred.dtype)
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            output_image = self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample.clamp(-1.0, 1.0)
        else:
            if c_t.shape[1] != self.conditioning_in_channels:
                raise ValueError(
                    f"SliceFixer expected {self.conditioning_in_channels} conditioning channels, "
                    f"got {c_t.shape[1]}."
                )
            c_t = self.input_adapter(c_t)
            # scale the lora weights based on the r value
            self.unet.set_adapters(["default"], weights=[r])
            set_weights_and_activate_adapters(self.vae, ["vae_skip"], [r])
            encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
            # combine the input and noise
            unet_input = encoded_control * r + noise_map * (1 - r)
            self.unet.conv_in.r = r
            unet_output = self.unet(unet_input, self.timesteps, encoder_hidden_states=caption_enc,).sample
            self.unet.conv_in.r = None
            x_denoised = self.sched.step(unet_output, self.timesteps, unet_input, return_dict=True).prev_sample
            x_denoised = x_denoised.to(unet_output.dtype)
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            self.vae.decoder.gamma = r
            output_image = self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample.clamp(-1.0, 1.0)
        return output_image

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}
        sd["conditioning_in_channels"] = self.conditioning_in_channels
        if not isinstance(self.input_adapter, torch.nn.Identity):
            sd["state_dict_input_adapter"] = self.input_adapter.state_dict()
        sd["use_xray_conditioning"] = self.use_xray_conditioning
        if self.use_xray_conditioning:
            sd["state_dict_fusion_adapter"] = self.fusion_adapter.state_dict()
        sd["intensity_domain"] = INTENSITY_DOMAIN
        sd["xray_conditioning"] = XRAY_CONDITIONING if self.use_xray_conditioning else None
        torch.save(sd, outf)
