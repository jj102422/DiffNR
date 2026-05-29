## Stage 1 — 低成本调参（零数据改动）详解

均为 `slicefixer/train_pix2pix_turbo.py` + 启动命令调整。**98be829 未实施这部分**，下面按"改什么 / 为什么 / 怎么验证 / 风险"逐项展开。

---

### 1.1 DataLoader 知识参数：`pin_memory` + `persistent_workers` + `prefetch_factor`

**改什么**（train DataLoader，原 `:333-338`；commit 后位于 `build_train_dataloader`）：
```python
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.train_batch_size,
    shuffle=True,
    num_workers=args.dataloader_num_workers,
    pin_memory=True,  # 新增
    persistent_workers=True,  # 新增（要求 num_workers > 0）
    prefetch_factor=4,  # 新增（默认 2）
)
```
val DataLoader 同理：`num_workers=0 → 4`，加 `persistent_workers=True`。

**为什么**：
- `pin_memory=True`：worker 把 batch 写入**锁页内存**（page-locked），GPU 可异步 DMA 拷贝；配合 `non_blocking=True` 才有效（见 1.3）。
- `persistent_workers=True`：worker 进程跨 epoch 复用，**避免每个 epoch 重启子进程**——SliceFixer 一个 epoch 有 数千 step、Dataset `__init__` 又要扫所有 case 拿 shape，重启代价大。
- `prefetch_factor=4`：每个 worker 预取 4 个 batch 而非默认 2，**掩盖** `.npz` 解压尖峰。
- `num_workers 2 → 8`（启动命令 `--dataloader_num_workers 8`）：每进程 8 worker × 4 进程 = 32 worker。**需先确认 CPU 核数**：`nproc` 看一下，超过物理核数会反向劣化（上下文切换 + 内存争抢）。

**怎么验证**：
- `time` 一个 epoch 头 200 step 的 wall time（先后对比）。
- `torch.profiler` 看 `enumerate(dl_train)` 时长占比，目标降到 <10%。
- `nvidia-smi dmon -s pucvmet -d 1` 看功率均值，目标 109W → 200W+。

**风险**：
- 32 worker × 每个 worker 自己的 `_mmap_cache` dict（commit 后引入的，`MedicalCTDataset:__init__`）+ `.npz` 解压时全卷在内存：**主机内存压力** 显著。`.npz` 数据如果是 512³ float32 = 512MB/卷，8 case/block × float32 ≈ 4GB/worker × 32 worker = 128GB——上限需要先看 `/proc/meminfo` 总内存。若爆，先把 `num_workers` 降到 4。
- `persistent_workers + use_volume_cache=True`：cache block 切换时 dataloader 也被销毁重建（`iter_epoch_batches` 里 `del block_loader`），persistent 实际生效域只在 block 内、跨 epoch 没好处。**低置信度**：此场景下 `persistent_workers` 收益有限。

---

### 1.2 增大 `train_batch_size`，相应调 `gradient_accumulation_steps`

**改什么**：启动命令 `--train_batch_size 1 → 4`，`--gradient_accumulation_steps 4 → 1`（保持 effective batch 不变）；若想放大 effective batch，保留 grad_accum=4 → effective batch 64（4 卡 × bs4 × ga4）。

**为什么**：
- 单步 SD-Turbo（单步 ADD 蒸馏）UNet 前向 + VAE 编解码计算量小，bs=1 时**每个 CUDA kernel launch 开销占比高**，SM 利用率拉不起来 → 功率低。
- 显存当前 14.8 / 24 GB，**约 9 GB 余量**。bs 翻 4 倍，激活 ≈ 4 倍线性增长，**最坏情况 14.8 + (14.8 - X) × 3** ≈ 24-32GB——会 OOM。安全策略：bs 1 → 2 先试，看显存再决定是否 → 4。
- 关键开销点：LPIPS(VGG)、CLIP（如果 `lambda_clipsim>0`）、CLIP-D 判别器（`lambda_gan>0`）都跟 batch 线性走，**这些是 bs↑ 时显存爆炸的主因，不是 UNet 本身**。可以临时 `lambda_clipsim=0` 拿到更高 bs 上限。
- `--gradient_checkpointing` 当前开启 → 用重算换显存，关掉它可省时间但更吃显存，bs↑ 通常与 checkpointing 共存。

**怎么验证**：
- 试 bs=2 跑 50 step 看峰值显存（`nvidia-smi` Mem 列或 `torch.cuda.max_memory_allocated()`）。
- 看 step/s 是否 < 4 倍提升（理想 bs×4 → step/s 同步 ×?，但因摊薄了 launch 开销，单步会 < ×4 慢，**等效 sample/s 应 > 1×**）。

**风险**：
- **学习率/收敛动力学**：lr_scheduler `num_training_steps` 已按 `num_processes` 缩放（commit 前的代码就有），但**没按 batch 缩放**。effective batch 16→64 时通常需 lr × 2 或预热步数延长，否则 loss 形态变化。属训练调参，无法机械改。
- LPIPS-VGG / CLIP-D 显存随 bs 线性，OOM 临界点要逐档试。
- val 不受影响（`dl_val` 固定 bs=1，且 commit 里在 eval 段没有 `assert B==1` 了，但 dataloader 仍是 1）。

---

### 1.3 `non_blocking=True` 拷贝（必须配合 `pin_memory=True`）

**改什么**：
```python
# train loop（commit 后 :462 附近）
x_src = batch["conditioning_pixel_values"].cuda(non_blocking=True)
x_tgt = batch["output_pixel_values"].cuda(non_blocking=True)
xray_feat1 = (
    batch["xray_feat1"].cuda(non_blocking=True) if args.use_xray_conditioning else None
)
xray_feat2 = (
    batch["xray_feat2"].cuda(non_blocking=True) if args.use_xray_conditioning else None
)
# eval loop 同样几处 .cuda() 加 non_blocking=True
```

**为什么**：默认 `.cuda()` 是**阻塞**的——CPU 等 H2D 拷贝完才继续。pin_memory + non_blocking 后，拷贝走异步 DMA，CPU 可继续启动后续 kernel，**重叠 H2D 与计算**。bs↑ 后拷贝量增大，收益更明显。

**怎么验证**：profiler 里 `aten::copy_` 与计算 kernel 是否在不同 stream 上并行。

**风险**：
- 紧跟着对该 tensor 做 CPU 端操作会读到未就绪数据。本代码里 `.cuda()` 后立刻进 GPU 前向，安全。
- `clip.tokenize(...).to(device)` 之类已经是 CPU 端 → 不在此列。

---

### 1.4 缓存常量 prompt token

**改什么**：循环外预算一次：
```python
# 进 train loop 前
constant_prompt_tokens = clip.tokenize([prompt], truncate=True).cuda()  # shape [1, 77]
constant_clip_text_features = None  # 可选，若 lambda_clipsim>0 也可缓存文本特征
```
train step 内（commit 后 `:496` 附近）：
```python
# 旧：
# caption_tokens = clip.tokenize(batch["caption"], truncate=True).to(x_tgt_pred.device)
# 新：
caption_tokens = constant_prompt_tokens.expand(B, -1)
```
eval 段同理。

**为什么**：prompt 是常量 `"high quality medical CT slice, ..."`（`:392`），每 step 在 CPU 上重 tokenize **完全是浪费**。`clip.tokenize` 内部走 BPE，是纯 Python，无 GPU 加速，会成为短 step 下的可见瓶颈。
- **更激进**：CLIP 文本 encoder 输出也是常量（同 prompt），可一次性 `net_clip.encode_text` 缓存，省掉每 step 的 text-encoder 前向。

**怎么验证**：profiler 里 `clip.tokenize` / text encoder 调用消失。

**风险**：极低。仅当 prompt 改成 per-sample 才需要回退。

---

### 1.5 降低 rank0 串行 stall

**改什么**（多选）：
- **已做（commit 98be829）**：`viz_freq / eval_freq` 默认 100 → 500。
- **未做**：`--checkpointing_steps` 调大（默认 1000，落盘到 `/root/epfs` 网络盘很慢）；或把 checkpoint 写到**本地盘** + 后台异步 rsync 到 `epfs`。
- **未做**：eval 已被 commit 改成多 rank 并行 → 这个 stall 源头已经消除。

**为什么**：rank0 进入 `is_main_process` 的串行段（log/viz/checkpoint）时，其它 rank 在 NCCL all-reduce 集合通信点 busy-wait spin → `nvidia-smi` 报 100% util 但功率低。频次降下来 + checkpoint 异步化 → 减少非对称 stall 时长占比。

**怎么验证**：临时把三个 freq 调到比 `max_train_steps` 还大（即不触发），重跑同样命令，看 GPU0 是否回到 ~100% util 且功率均衡。

**风险**：
- `checkpointing_steps` 调太大 → 崩溃后丢的训练步数多。
- 本地盘异步 rsync：磁盘容量 + crash 时未同步数据丢失。

---

### Stage 1 实施顺序建议（按时间成本由小到大）

| 优先级 | 项                            | 代码改动量    | 风险                    |
| ------ | ----------------------------- | ------------- | ----------------------- |
| 1      | 1.4 缓存 prompt token         | ~5 行         | 极低                    |
| 2      | 1.1 DataLoader 4 个参数       | ~4 行         | 低（注意内存）          |
| 3      | 1.3 `non_blocking=True`       | ~6 处         | 低                      |
| 4      | 1.5 调 freq / checkpoint 异步 | 参数 / ~10 行 | 低                      |
| 5      | 1.2 batch_size↑               | 启动参数      | **中**（OOM + lr 重调） |

每项做完跑 100~200 step、记录 step/s 与峰值显存与平均功率，再决定是否继续往下做。先做 1~4 拿低成本收益，**最后**才动 batch（因为有 OOM 和训练动力学风险）。

要我开始动手吗？建议先 1+2+3+4 一次性补上（这四项无相互依赖且都低风险），batch 留给你单独决定。