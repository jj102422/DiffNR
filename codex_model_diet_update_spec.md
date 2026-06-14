# 多模型 CT 重建评估 Pipeline 修改说明：Model Diet 配置与 GT-space 对齐

本文档用于交给 Codex 修改当前 evaluation pipeline。目标是把 x2ct、PerX2CT、DIF、NAF、3DGS、DiffNR/SliceFixer 系列模型的重建结果统一读入、按模型专属 Diet 配置进行强度和空间对齐，然后调用同一个 `metric(GT, pred, mask, Diet)` 计算 MAE、PSNR、SSIM、LPIPS，并输出 test set 的 per-case 与 summary 结果。

---

## 0. 核心目标

当前已有多个模型的重建结果：

- x2ct
- PerX2CT
- DIF-Net
- NAF
- raw 3DGS
- 3DGS + SliceFixer Post
- 3DGS + SliceFixer Iterative
- PerX2CT + SliceFixer No-mask Post
- PerX2CT + SliceFixer Mask Full Ckpt

这些模型的输出空间、轴顺序、强度域、文件格式、是否带 spacing/origin/direction 都不同。因此不能直接把 prediction resize 到 GT shape 后算指标。需要在 `prepare_case_for_metric(case_id, model_name, diet)` 中根据每个模型的 Diet 配置完成以下操作：

```text
load prediction
→ squeeze
→ transpose / flip
→ inverse intensity
→ spatial align to selected GT/reference grid
→ clip to canonical evaluation intensity range
→ return GT, pred, mask, debug_info
→ metric(GT, pred, mask, Diet)
```

正式指标要求：

- MAE：在 GT mask 内 voxel-wise 计算。
- PSNR：在 GT mask 内基于 MSE 计算。
- SSIM：slice-wise 计算 SSIM map，然后在 mask 内平均。
- LPIPS：slice-wise，使用 mask bbox crop 后计算。
- 汇总：每个 case 先得到一个指标值，然后对 test set 求 mean/std。

---

## 1. 代码需要实现或修改的模块

建议目录结构如下：

```text
evaluation/
├── configs/
│   ├── model_diet.yaml
│   └── eval_global.yaml
├── scripts/
│   ├── evaluate_all.py
│   ├── verify_alignment.py
│   └── build_three_plane_debug.py
├── utils/
│   ├── io_utils.py
│   ├── intensity_utils.py
│   ├── spatial_utils.py
│   ├── metric_utils.py
│   └── debug_utils.py
└── outputs/
    ├── per_case_metrics.csv
    ├── summary_metrics.csv
    ├── debug_alignment.csv
    ├── failed_cases.csv
    └── three_plane_compare/
```

需要重点实现：

```python
def prepare_case_for_metric(case_id: str, model_name: str, diet: dict, global_cfg: dict):
    """Read GT, mask and pred; convert pred to the selected evaluation/reference grid."""


def metric(gt, pred, mask, diet: dict, global_cfg: dict):
    """Compute MAE, PSNR, SSIM, LPIPS. Assumes gt/pred/mask already share the same shape."""
```

注意：`metric()` 不负责模型专属的文件读取、transpose、flip、反归一化和 resize；这些都必须在 `prepare_case_for_metric()` 中完成。

---

## 2. 全局评估配置建议

新建 `configs/eval_global.yaml`：

```yaml
eval:
  # 建议先固定一个全局参考空间。若使用 original CT GT，则所有模型 pred 都重采样到这个空间。
  # 若某些模型只能严格和 processed GT 比较，则 alignment_quality 必须标为 approximate 或 verified_processed_only。
  reference_space: "canonical_gt"

  gt:
    loader: "project_specific"
    axis_order_after_loading: "ZYX"
    shape_source: "GT.shape"

  mask:
    source: "GT mask"
    binarize_threshold: 0
    interpolation: "nearest"

  intensity:
    # 根据当前数据选择一种统一评估强度空间。
    # 可选："hu_like" 或 "shifted_ct"。
    # 如果 GT 是原始 HU，则 pred 必须映射到 HU-like 后再算。
    # 如果 GT 是 H5 训练流的 shifted/normalized CT，则 pred 和 GT 应统一到 shifted CT 后再算。
    canonical_space: "project_defined"
    clip_for_metric: [-1024, 3000]
    norm_for_psnr_ssim_lpips: "minmax_by_fixed_range"
    norm_min: -1024
    norm_max: 3000

  spatial:
    default_pred_interpolation: "linear"
    default_mask_interpolation: "nearest"
    fill_value_default: -1024

  metrics:
    compute_mae: true
    compute_psnr: true
    compute_ssim: true
    compute_lpips: true
    ssim:
      slice_axis: 0
      data_range: 1.0
      min_mask_pixels_per_slice: 100
    lpips:
      slice_axis: 0
      min_mask_pixels_per_slice: 100
      bbox_margin: 8
      resize_hw: [256, 256]
      input_range: [-1, 1]
      replicate_gray_to_rgb: true
      net: "alex"

  debug:
    save_debug_csv: true
    save_three_plane_png: true
    window_vmin: -1024
    window_vmax: 1000
    warn_if_pred_p1_greater_than: -500
    warn_if_shape_mismatch: true
```

重要说明：

1. 当前各模型的 inverse formula 并不完全处于同一个医学 HU 空间。有些是 `clip(pred,0,1)*2500`，有些是 `clip(pred,0,1)*2500-1024`，有些是 `*3000`。代码必须允许每个模型独立 inverse，但最终进入 metric 前，GT 和 pred 必须在同一个 `canonical_space`。
2. 如果暂时不能把某模型严格映射回原始 CT 物理 FOV，需要将 `alignment_quality` 标记为 `approximate`，并在 summary 表中保留该字段。
3. 不要用每个 case 自己的 min-max 做归一化，这会让指标虚高。PSNR/SSIM/LPIPS 使用固定强度范围归一化。

---

## 3. Model Diet YAML 模板

`configs/model_diet.yaml` 中每个模型采用统一 schema：

```yaml
MODEL_NAME:
  enabled: true

  # A. 文件读取
  pred_path_pattern: ""
  pred_format: "npy | npz | nii.gz | mha"
  read_backend: "numpy | nibabel | SimpleITK"
  pred_key: null

  # B. 模型原始空间
  model_space: ""
  geometry_source: ""
  pred_raw_shape: "read_from_file"
  pred_axis_order_in_file: "ZYX | XYZ | DHW | ..."
  spacing_mm: null
  volume_extent_mm: null
  origin_direction_available: false

  # C. 轴向修正
  squeeze_axes: []
  transpose_order: [0, 1, 2]
  flip_axes: []

  # D. 强度修正
  output_domain: ""
  inverse_intensity:
    type: "identity | affine | project_specific"
    formula: ""
    clip_hu: null
  fill_value_hu: -1024

  # E. 空间对齐
  spatial_align:
    method: "identity | resize_to_gt_shape | crop_pad_resize | center_crop_or_pad_z"
    interpolation: "linear"
    target_shape: "GT.shape"
    target_spacing: "GT.spacing"
    mask_interpolation: "nearest"

  # F. 评估控制
  allow_metric: true
  alignment_quality: "verified | approximate | verified_processed_only | unverified"
  notes: ""
```

---

## 4. 需要写入 `model_diet.yaml` 的模型配置

下面配置是当前项目状态下的 Diet 初稿。Codex 应按这些字段修改代码，使 pipeline 能够逐模型读取、转换、对齐和 debug。

### 4.1 PerX2CT

```yaml
PerX2CT:
  enabled: true

  # A. 文件读取
  pred_path_pattern: "<save_dir>/<patient>/<axis>_<slice_idx:03d>.npz"
  pred_format: "npz"
  read_backend: "numpy"
  pred_key: "vol_pred"

  # B. 模型原始空间
  model_space: "cropped-space -> assembled GT voxel grid"
  geometry_source: "plastimatch DRR + preprocessed original_CT voxel grid"
  pred_raw_shape: "decoder patch: 128x128; saved slice: 512x512; volume: Zx512x512 when stacking axial_*.npz"
  pred_axis_order_in_file: "DHW when stacking saved axial slices in filename order"
  spacing_mm: null
  volume_extent_mm: null
  origin_direction_available: false

  # C. 轴向修正
  squeeze_axes: []
  transpose_order: [0, 1, 2]
  flip_axes: []

  # D. 强度修正
  output_domain: "norm_0_1"
  inverse_intensity:
    type: "affine"
    formula: "pred_shifted_ct = pred * 2500.0 + 0.0; pred_hu = pred * 2500.0 - 1000.0 only if GT is original HU after +1000 preprocessing"
    clip_hu: null
  fill_value_hu: -1000

  # E. 空间对齐
  spatial_align:
    method: "crop_pad_resize"
    interpolation: "none for 128 patch tiling; linear only if an external 320->512 resize is added"
    target_shape: "GT.shape, normally Zx512x512"
    target_spacing: null
    mask_interpolation: "nearest"

  # F. 评估控制
  allow_metric: true
  alignment_quality: "approximate"
  notes: >
    Current runnable path uses 512 CT with 128 grid patches and stitches 4x4 back to 512.
    The old 320 config has no verified 320->512 recovery path in this code; do not directly
    compare 320 outputs to 512 GT without explicit resampling and orientation checks.
    According to current default main_test.py + configs/PerX2CT_global_w_zoomin.yaml:
    output/GT training domain is [0,1] normalized CT, not [-1,1]. The decoder ends with conv_out,
    without forced sigmoid/tanh, so values may slightly exceed [0,1], but semantic domain is norm_0_1.
    The save script does not inverse-normalize; it saves vol_pred. To recover training CT values:
    pred * 2500 + 0. If GT is original HU and data was shifted by +1000, use pred_hu = pred * 2500 - 1000.
    There is crop/zoom mechanism. Training uses random crop; test uses 128 patch grid crop.
    main_test_zoom.py has a separate zoom-in evaluation path.
```

Implementation notes for Codex:

- Add a loader that stacks `axial_*.npz` in filename order into `(D,H,W)`.
- If values are slightly outside `[0,1]`, use `clip_before_inverse: true` as an optional parameter or warning.
- Keep an explicit switch between shifted CT evaluation and HU-like evaluation.

---

### 4.2 NAF

```yaml
NAF:
  enabled: true

  # A. 文件读取
  pred_path_pattern: "/home/public/CTSpine1K/data/naf_result/{case}/image_pred.nii.gz"
  pred_format: "nii.gz"
  read_backend: "SimpleITK"
  pred_key: null

  # B. 模型原始空间
  model_space: "resized-space / TIGRE-space"
  geometry_source: "TIGRE"
  pred_raw_shape: "read_from_file"
  pred_axis_order_in_file: "ZYX"
  spacing_mm: [1.0, 1.0, 1.0]
  volume_extent_mm: "[256, 256, Z]  # Z varies by case; equals nVoxel because dVoxel=[1,1,1]"
  origin_direction_available: true

  # C. 轴向修正
  squeeze_axes: []
  transpose_order: [0, 1, 2]
  flip_axes: []

  # D. 强度修正
  output_domain: "norm_0_1"
  inverse_intensity:
    type: "affine"
    formula: "pred_hu = clip(pred, 0, 1) * 2500.0"
    clip_hu: [0, 2500]
  fill_value_hu: 0

  # E. 空间对齐
  spatial_align:
    method: "resize_to_gt_shape"
    interpolation: "linear"
    target_shape: "GT.shape"
    target_spacing: "GT.spacing"
    mask_interpolation: "nearest"

  # F. 评估控制
  allow_metric: true
  alignment_quality: "approximate"
  notes: >
    NAF output is reconstructed in resized TIGRE space, not original CT space.
    Original GT was converted from MHA ZYX to internal XYZ, resized to X/Y=256 with isotropic 1mm voxels,
    normalized by clip(raw,0,inf)/2500, then projected by TIGRE.
    If evaluating against original ct_file.mha without modifying GT, read pred by SimpleITK as ZYX,
    clip to [0,1], map to raw scale by *2500, then resize pred to GT.shape. Do not crop Z.
    This alignment is approximate because inverse resize does not restore original acquisition geometry exactly.
```

Implementation notes for Codex:

- Read by SimpleITK gives numpy array in `(Z,Y,X)`.
- Do not crop Z unless explicitly configured.
- Because `fill_value_hu` is `0`, visualization in HU window may look gray if GT is original HU. This is expected only when evaluating in shifted CT space. If global `canonical_space` is original HU-like, convert consistently or flag warning.

---

### 4.3 raw 3DGS

```yaml
raw_3DGS:
  enabled: true

  # A. 文件读取
  pred_path_pattern: "/root/epfs/test/01_raw_3dgs/{case_id}/point_cloud/iteration_12000/vol_pred.npy"
  pred_format: "npy"
  read_backend: "numpy"
  pred_key: null

  # B. 模型原始空间
  model_space: "TIGRE-space"
  geometry_source: "TIGRE"
  pred_raw_shape: "read_from_file"
  pred_axis_order_in_file: "XYZ"
  spacing_mm: null
  volume_extent_mm: "from scanner_cfg during generation; not stored in pred file"
  origin_direction_available: false

  # C. 轴向修正
  squeeze_axes: []
  transpose_order: [2, 1, 0]
  flip_axes: []

  # D. 强度修正
  output_domain: "norm_0_1"
  inverse_intensity:
    type: "affine"
    formula: "pred_hu = clip(pred, 0, 1) * 3000"
    clip_hu: [0, 3000]
  fill_value_hu: -1024

  # E. 空间对齐
  spatial_align:
    method: "crop_pad_resize / center_crop_or_pad_z"
    interpolation: "linear"
    target_shape: "GT.shape"
    target_spacing: "GT.spacing"
    mask_interpolation: "nearest"

  # F. 评估控制
  allow_metric: true
  alignment_quality: "approximate"
  notes: >
    vol_pred.npy is saved by Scene.save in train_DiffNR.py.
    The file itself does not store spacing/origin/direction.
```

Implementation notes for Codex:

- Apply `transpose_order: [2,1,0]` to convert `XYZ` file order into `ZYX` array order for GT comparison.
- Since this is TIGRE-space without stored physical metadata, formal metric should retain `alignment_quality=approximate`.

---

### 4.4 3DGS + SliceFixer Iterative

```yaml
3DGS_SliceFixer_iterative:
  enabled: true

  # A. 文件读取
  pred_path_pattern: "/root/epfs/test/03_3dgs_slicefixer_iter/{case_id}/point_cloud/iteration_12000/vol_pred.npy"
  pred_format: "npy"
  read_backend: "numpy"
  pred_key: null

  # B. 模型原始空间
  model_space: "TIGRE-space"
  geometry_source: "TIGRE"
  pred_raw_shape: "read_from_file"
  pred_axis_order_in_file: "XYZ"
  spacing_mm: null
  volume_extent_mm: "from scanner_cfg during generation; not stored in pred file"
  origin_direction_available: false

  # C. 轴向修正
  squeeze_axes: []
  transpose_order: [2, 1, 0]
  flip_axes: []

  # D. 强度修正
  output_domain: "norm_0_1 / SliceFixer-compatible normalized volume"
  inverse_intensity:
    type: "affine"
    formula: "pred_hu = inverse_slicefixer_output(pred) = clip(pred, 0, 1) * 3000"
    clip_hu: [0, 3000]
  fill_value_hu: -1024

  # E. 空间对齐
  spatial_align:
    method: "crop_pad_resize / center_crop_or_pad_z"
    interpolation: "linear"
    target_shape: "GT.shape"
    target_spacing: "GT.spacing"
    mask_interpolation: "nearest"

  # F. 评估控制
  allow_metric: true
  alignment_quality: "approximate"
  notes: >
    This model continues training from raw 3DGS chkpnt10000.pth.
    SliceFixer is used as diffusion target/loss during iterative optimization.
    The final saved file is still vol_pred.npy generated by train_DiffNR.py.
```

---

### 4.5 PerX2CT + SliceFixer No-mask Post

```yaml
PerX2CT_SliceFixer_nomask_post:
  enabled: true

  # A. 文件读取
  pred_path_pattern: "/root/epfs/test/04_perx2ct_slicefixer_nomask_post/{case_id}/vol_pred_slicefixer.npz"
  pred_format: "npz"
  read_backend: "numpy"
  pred_key: "vol_pred"

  # B. 模型原始空间
  model_space: "cropped-space / CT slice stack space"
  geometry_source: "original_CT"
  pred_raw_shape: "read_from_file"
  pred_axis_order_in_file: "XYZ"
  spacing_mm: null
  volume_extent_mm: "not stored in pred file"
  origin_direction_available: false

  # C. 轴向修正
  squeeze_axes: []
  transpose_order: [2, 1, 0]
  flip_axes: []

  # D. 强度修正
  output_domain: "norm_0_1"
  inverse_intensity:
    type: "affine"
    formula: "pred_hu = clip(pred, 0, 1) * 2500"
    clip_hu: [0, 2500]
  fill_value_hu: -1024

  # E. 空间对齐
  spatial_align:
    method: "crop_pad_resize / center_crop_or_pad_z"
    interpolation: "linear"
    target_shape: "GT.shape"
    target_spacing: "GT.spacing"
    mask_interpolation: "nearest"

  # F. 评估控制
  allow_metric: true
  alignment_quality: "approximate"
  notes: >
    scripts/slicefixer_postprocess_case.py reads case/pred/axial_*.npz,
    sends each slice to SliceFixer, stacks output and saves vol_pred_slicefixer.npz.
    This variant uses no mask conditioning.
```

---

### 4.6 DIF-Net CTSpine1K Current

```yaml
DIF_Net_CTSpine1K_current:
  enabled: true

  # A. 文件读取
  pred_path_pattern: "/home/public/CTSpine1K/data/DIF_result（1）/{name}.nii.gz"
  pred_format: "nii.gz"
  read_backend: "nibabel"
  pred_key: null

  # B. 模型原始空间
  model_space: "GT-space"
  geometry_source: "plastimatch_xray + config.yaml(TIGRE-style), not original_CT"
  pred_raw_shape: "read_from_file; expected (512, 512, z_length[name])"
  pred_axis_order_in_file: "XYZ"
  spacing_mm: [1.0, 1.0, 1.0]
  volume_extent_mm: "[512, 512, z_length[name]] under current header/config; original extent unknown"
  origin_direction_available: false

  # C. 轴向修正
  squeeze_axes: []
  transpose_order: [0, 1, 2]
  flip_axes: []

  # D. 强度修正
  output_domain: "hu"
  inverse_intensity:
    type: "affine"
    formula: "pred_hu = clip(pred_norm * 2500.0 + 0.0, 0, 3000)"
    clip_hu: [0, 3000]
  fill_value_hu: 0

  # E. 空间对齐
  spatial_align:
    method: "identity"
    interpolation: "linear"
    target_shape: "processed_H5_GT.transpose(2,1,0).shape == (512,512,z_length[name])"
    target_spacing: [1.0, 1.0, 1.0]
    mask_interpolation: "nearest"

  # F. 评估控制
  allow_metric: true
  alignment_quality: "verified"
  notes: >
    Metric is valid only against the current processed H5 GT after transpose to (X,Y,Z).
    It is not verified for original DICOM/raw GT physical FOV.
    To map back to original GT FOV, need original CT affine/spacing/direction plus crop/resample offsets;
    these are not present in current H5 attrs or pred NIfTI.
    Current CTSpine1K/H5 training flow is not crop/resize to fixed cube. It reads H5 ct,
    converts original (Z,Y,X) to (X,Y,Z), and keeps variable Z.
    Only the original data/knee_cbct example flow crops/pads to 256^3.
    pred is not a low-resolution canonical volume. Eval generates a full grid of (512,512,real_z),
    model predicts point-wise and reshapes to image.shape.
    It cannot reliably map back to original GT FOV because H5 and pred NIfTI do not store
    original spacing/origin/direction/crop offset; pred is saved with default identity affine.
```

Implementation notes for Codex:

- For DIF, `identity` means identity only when GT is the processed H5 grid transposed to `(X,Y,Z)`.
- If the global GT loader returns `(Z,Y,X)`, either transpose DIF to `(Z,Y,X)` or transpose GT/mask to `(X,Y,Z)` consistently before metric. Do not silently compare mismatched axis orders.

---

### 4.7 x2ct

```yaml
x2ct:
  enabled: true

  # A. 文件读取
  pred_path_pattern: "/home/public/CTSpine1K/data/x2ct_result/multiview_test/CTSpine1K_200.1epoch/valid--tag=d2_multiview2500_90/CT/*/fake_ct.mha"
  pred_format: "mha"
  read_backend: "SimpleITK"
  pred_key: null

  # B. 模型原始空间
  model_space: "resized-space"
  geometry_source: "original_CT_preprocessed_h5"
  pred_raw_shape: [128, 128, 128]
  pred_axis_order_in_file: "ZYX"
  spacing_mm: [1.0, 1.0, 1.0]
  volume_extent_mm: [128.0, 128.0, 128.0]
  origin_direction_available: true

  # C. 轴向修正
  squeeze_axes: []
  transpose_order: [0, 1, 2]
  flip_axes: []

  # D. 强度修正
  output_domain: "hu"
  inverse_intensity:
    type: "affine"
    formula: "pred_hu = clip(pred_norm, 0, 1) * 2500 - 1024"
    clip_hu: [-1024, 1476]
  fill_value_hu: -1024

  # E. 空间对齐
  spatial_align:
    method: "identity"
    interpolation: "linear"
    target_shape: "GT.shape"
    target_spacing: "GT.spacing"
    mask_interpolation: "nearest"

  # F. 评估控制
  allow_metric: true
  alignment_quality: "verified"
  notes: >
    fake_ct.mha and real_ct.mha saved by visual.py are in the same resized 128^3 space.
    visual.py applies pred_01 * 2500 - 1024 before saving .mha; observed sample range is [-1024,1476].
    However, this is not strict original HU because source .h5['ct'] is clipped/normalized by [0,2500]
    before subtracting 1024.
    fake_ct.mha and saved real_ct.mha have the same shape, observed as (128,128,128).
    For current align_ct_xray_views_std output, no extra transpose is needed; SimpleITK reads numpy array as ZYX.
    saved .mha does not need an additional flip. visual.py already applied depth flip before saving: [:, ::-1, :, :].
    If evaluating against original CT/original HU, current spacing/origin/direction is insufficient and the result is only approximate.
```

Implementation notes for Codex:

- If using `identity`, the GT for x2ct should be its corresponding saved `real_ct.mha` or an equivalent 128^3 processed GT.
- If global evaluation uses original CT GT, change `spatial_align.method` to `resize_to_gt_shape` and mark `alignment_quality=approximate`.

---

### 4.8 3DGS + SliceFixer Post

```yaml
3DGS_SliceFixer_post:
  enabled: true

  # A. 文件读取
  pred_path_pattern: "/root/epfs/test/02_3dgs_slicefixer_post/{case_id}/vol_pred_slicefixer.npy"
  pred_format: "npy"
  read_backend: "numpy"
  pred_key: null

  # B. 模型原始空间
  model_space: "TIGRE-space"
  geometry_source: "TIGRE"
  pred_raw_shape: "read_from_file"
  pred_axis_order_in_file: "XYZ"
  spacing_mm: null
  volume_extent_mm: "from raw 3DGS scanner_cfg; not stored in pred file"
  origin_direction_available: false

  # C. 轴向修正
  squeeze_axes: []
  transpose_order: [2, 1, 0]
  flip_axes: []

  # D. 强度修正
  output_domain: "norm_0_1 / SliceFixer output normalized volume"
  inverse_intensity:
    type: "affine"
    formula: "pred_hu = inverse_slicefixer_output(pred) = clip(pred, 0, 1) * 3000"
    clip_hu: [0, 3000]
  fill_value_hu: -1024

  # E. 空间对齐
  spatial_align:
    method: "crop_pad_resize / center_crop_or_pad_z"
    interpolation: "linear"
    target_shape: "GT.shape"
    target_spacing: "GT.spacing"
    mask_interpolation: "nearest"

  # F. 评估控制
  allow_metric: true
  alignment_quality: "approximate"
  notes: >
    scripts/postprocess_volume_with_slicefixer.py reads raw 3DGS vol_pred.npy,
    sends each axial slice to SliceFixer, stacks output, and saves both vol_pred_slicefixer.npy
    and vol_pred_slicefixer.npz. The evaluation config uses .npy.
```

---

### 4.9 PerX2CT + SliceFixer Mask Full Ckpt

```yaml
PerX2CT_SliceFixer_mask_full_ckpt:
  enabled: true

  # A. 文件读取
  pred_path_pattern: "/root/epfs/test/05_perx2ct_slicefixer_mask_post/{case_id}/vol_pred_slicefixer.npz"
  pred_format: "npz"
  read_backend: "numpy"
  pred_key: "vol_pred"

  # B. 模型原始空间
  model_space: "cropped-space / CT slice stack space"
  geometry_source: "original_CT"
  pred_raw_shape: "read_from_file"
  pred_axis_order_in_file: "XYZ"
  spacing_mm: null
  volume_extent_mm: "not stored in pred file"
  origin_direction_available: false

  # C. 轴向修正
  squeeze_axes: []
  transpose_order: [2, 1, 0]
  flip_axes: []

  # D. 强度修正
  output_domain: "norm_0_1"
  inverse_intensity:
    type: "affine"
    formula: "pred_hu = clip(pred, 0, 1) * 2500"
    clip_hu: [0, 2500]
  fill_value_hu: -1024

  # E. 空间对齐
  spatial_align:
    method: "crop_pad_resize / center_crop_or_pad_z"
    interpolation: "linear"
    target_shape: "GT.shape"
    target_spacing: "GT.spacing"
    mask_interpolation: "nearest"

  # F. 评估控制
  allow_metric: true
  alignment_quality: "approximate"
  notes: >
    scripts/slicefixer_postprocess_case.py reads case/pred/axial_*.npz and additionally concatenates
    binary mask slices as conditioning. It then stacks output and saves vol_pred_slicefixer.npz.
    During evaluation, mask conditioning is not used again; only the final predicted volume is loaded.
```

---

## 5. 必须实现的文件读取逻辑

### 5.1 通用读文件函数

```python
def load_prediction_by_diet(case_id: str, model_name: str, model_cfg: dict) -> np.ndarray:
    """Load model prediction according to pred_format/read_backend/pred_key."""
```

需要支持：

1. `npy`：`np.load(path)`。
2. `npz`：`np.load(path)[pred_key]`。
3. `nii.gz` + `SimpleITK`：`sitk.GetArrayFromImage(sitk.ReadImage(path))`，返回通常为 `(Z,Y,X)`。
4. `nii.gz` + `nibabel`：`nib.load(path).get_fdata()`，通常保持 NIfTI data array order，需要按 diet 中 axis_order 处理。
5. `mha` + `SimpleITK`：同 SimpleITK，返回 `(Z,Y,X)`。
6. PerX2CT 原始 patch/slice 输出：需要根据 `<axis>_<slice_idx>.npz` 收集 `axial_*.npz` 并 stack 成 volume。

### 5.2 路径 pattern

实现：

```python
path = pred_path_pattern.format(case_id=case_id, case=case_id, name=case_id)
```

若 pattern 中包含 `*`，使用 `glob.glob`。如果匹配多个结果，按字典序排序并给出 warning；通常应只匹配一个。

---

## 6. 必须实现的轴顺序和方向处理

处理顺序：

```python
pred = np.asarray(pred)
pred = squeeze_axes(pred, cfg["squeeze_axes"])
pred = np.transpose(pred, cfg["transpose_order"])
for axis in cfg["flip_axes"]:
    pred = np.flip(pred, axis=axis)
```

注意：

- 对 `pred_axis_order_in_file: XYZ` 且目标 GT array 为 `ZYX` 的模型，通常使用 `transpose_order: [2,1,0]`。
- 对 SimpleITK 读出的 `.mha/.nii.gz`，通常已经是 `ZYX`。
- 不要因为 shape 相同就跳过 transpose/flip。
- 每次处理后记录到 `debug_alignment.csv`。

---

## 7. 必须实现的强度反变换

实现通用函数：

```python
def inverse_intensity(pred: np.ndarray, model_cfg: dict, global_cfg: dict) -> np.ndarray:
    """Convert model output to the configured intensity space."""
```

必须支持这些 formula：

```text
clip(pred, 0, 1) * 2500
clip(pred, 0, 1) * 2500 - 1000
clip(pred_norm, 0, 1) * 2500 - 1024
clip(pred, 0, 1) * 3000
clip(pred_norm * 2500.0 + 0.0, 0, 3000)
identity
```

建议不要直接 `eval(formula)`。用结构化字段更安全，但当前 formula 已经是字符串，至少要通过 if/elif 或白名单解析。

建议增加可选字段：

```yaml
inverse_intensity:
  clip_before_inverse: true
  scale: 2500.0
  offset: 0.0
  output_clip: [0, 2500]
```

内部用：

```python
pred = np.clip(pred, 0, 1) if clip_before_inverse else pred
pred = pred * scale + offset
pred = np.clip(pred, output_clip[0], output_clip[1]) if output_clip else pred
```

---

## 8. 必须实现的空间对齐方法

实现：

```python
def align_prediction_to_gt(pred, gt, model_cfg, global_cfg):
    method = model_cfg["spatial_align"]["method"]
```

至少支持：

### 8.1 identity

只检查 shape 是否一致：

```python
if pred.shape != gt.shape:
    raise ValueError or warning + fallback if explicitly allowed
```

### 8.2 resize_to_gt_shape

使用线性插值把 pred resize 到 `gt.shape`。

```python
from scipy.ndimage import zoom
zoom_factors = [g / p for g, p in zip(gt.shape, pred.shape)]
pred_resized = zoom(pred, zoom_factors, order=1)
```

### 8.3 crop_pad_resize / center_crop_or_pad_z

适用于 3DGS、SliceFixer stack 等。建议先中心 crop/pad 到接近 GT 的 Z，再对全体 resize 到 GT shape。

基本逻辑：

```python
pred = center_crop_or_pad_to_shape(pred, target_shape=gt.shape, fill_value=fill_value_hu)
if pred.shape != gt.shape:
    pred = resize_to_shape(pred, gt.shape, order=1)
```

注意：

- `fill_value_hu` 必须来自模型 Diet。
- 如果 GT 是 HU-like，空气填充值通常为 `-1024` 或 `-1000`。
- 如果 GT 是 shifted CT，空气填充值可能为 `0`。
- 不要默认填 0，否则 HU window 可视化会发灰。

---

## 9. GT 和 mask 处理

GT 和 mask 必须由统一 loader 提供：

```python
gt = load_gt(case_id, global_cfg)
mask = load_gt_mask(case_id, global_cfg)
```

要求：

1. `gt.shape == mask.shape`。
2. `mask = mask > threshold`。
3. 如果 mask 需要 resample，必须用 nearest neighbor。
4. 如果某个模型的严格参考空间不是 canonical GT，而是 processed GT，例如 x2ct 的 saved real_ct.mha 或 DIF 的 processed H5 grid，代码必须显式选择对应 reference GT，并在 debug 中记录 `reference_gt_source`。

---

## 10. 指标计算要求

### 10.1 MAE

在 mask 内计算：

```python
mae = np.mean(np.abs(pred[mask] - gt[mask]))
```

建议同时输出：

- `MAE_raw_space`
- `MAE_norm_space`

但主表至少保留一个固定定义。

### 10.2 PSNR

先用固定范围归一化：

```python
gt_norm = (gt - norm_min) / (norm_max - norm_min)
pred_norm = (pred - norm_min) / (norm_max - norm_min)
gt_norm = np.clip(gt_norm, 0, 1)
pred_norm = np.clip(pred_norm, 0, 1)
```

然后 mask 内计算 MSE：

```python
mse = np.mean((pred_norm[mask] - gt_norm[mask]) ** 2)
psnr = 10 * np.log10(1.0 / mse)
```

若 `mse == 0`，返回 `np.inf`。

### 10.3 SSIM

slice-wise 计算 SSIM map，然后 mask 内平均：

```python
ssim_value, ssim_map = structural_similarity(
    gt_slice_norm,
    pred_slice_norm,
    data_range=1.0,
    full=True,
)
slice_score = ssim_map[mask_slice].mean()
```

要求：

- 跳过 `mask_slice.sum() < min_mask_pixels_per_slice` 的 slice。
- 最终 case-level SSIM 为有效 slice 的平均值。
- 不要把 mask 外置 0 后直接整图 SSIM，因为背景会主导结果。

### 10.4 LPIPS

slice-wise，使用 mask bbox crop：

1. 对每个有效 slice 取 `mask_slice` 的 bbox。
2. bbox 外扩 `bbox_margin`。
3. crop GT 和 pred。
4. resize 到 `256x256`。
5. 单通道复制成 3 通道。
6. `[0,1]` 转 `[-1,1]`。
7. 输入 LPIPS 网络。
8. 对有效 slice 求平均。

如果 case 没有有效 slice，返回 NaN 并写入 warning。

---

## 11. Debug 输出要求

### 11.1 debug_alignment.csv

每个 model-case 记录：

```text
model
case_id
reference_gt_source
gt_shape
mask_shape
pred_raw_shape
pred_after_axis_shape
pred_final_shape
gt_min
gt_max
gt_p1
gt_p99
pred_min
pred_max
pred_p1
pred_p99
mask_voxels
output_domain
inverse_intensity_formula
fill_value_hu
transpose_order
flip_axes
spatial_align_method
allow_metric
alignment_quality
warnings
```

### 11.2 failed_cases.csv

记录：

```text
model
case_id
stage
error_message
```

### 11.3 three-plane compare PNG

继续保留当前三平面对比图，建议每个模型显示：

```text
model_name
p1-p99
shape
transpose_order
inverse_formula
alignment_quality
```

所有模型必须使用统一 CT window，例如：

```python
vmin = -1024
vmax = 1000
```

如果某些模型在 shifted CT 空间评估，可以增加另一套 window 或在标题注明 `intensity_space=shifted_ct`。

---

## 12. 正式输出表

### 12.1 per_case_metrics.csv

```text
model
case_id
MAE
PSNR
SSIM
LPIPS
mask_voxels
alignment_quality
allow_metric
warnings
```

### 12.2 summary_metrics.csv

每个模型输出 mean/std/count：

```text
model
num_cases
MAE_mean
MAE_std
PSNR_mean
PSNR_std
SSIM_mean
SSIM_std
LPIPS_mean
LPIPS_std
alignment_quality
notes
```

汇总方式必须是：

```text
先每个 case 算一个指标，再对 test cases 求平均。
```

不要把所有病例的 voxel 合并后一次性计算总指标。

---

## 13. 必须加的安全检查

在 `prepare_case_for_metric()` 中加入：

```python
assert gt.ndim == 3
assert pred.ndim == 3
assert mask.ndim == 3
assert gt.shape == pred.shape == mask.shape
assert mask.dtype == bool
```

软性 warning：

```python
if pred_p1 > global_cfg["eval"]["debug"]["warn_if_pred_p1_greater_than"]:
    warnings.append("pred_p1 is too high; background may not match GT intensity space")

if abs(pred_p1 - gt_p1) > 800:
    warnings.append("pred/GT intensity domain may be mismatched")

if pred.std() < small_threshold:
    warnings.append("prediction may be over-smoothed or nearly constant")
```

---

## 14. 当前配置的关键风险提示

1. NAF、DIF、PerX2CT、3DGS/SliceFixer 系列很多 inverse formula 是 `*2500` 或 `*3000`，没有减去空气 offset。如果全局 GT 是原始 HU，这些模型背景会发灰，需要统一转换到 HU-like 或改用 shifted CT GT。
2. x2ct 的 `identity` 只对 saved fake_ct.mha 与 saved real_ct.mha 的 128^3 processed space 成立。如果和原始 CT GT 比较，应改为 approximate resampling。
3. DIF 的 `identity` 只对 processed H5 GT transpose 后的 `(X,Y,Z)` 成立，不代表原始 DICOM/raw GT physical FOV。
4. raw 3DGS 和 3DGS SliceFixer 没有 spacing/origin/direction，只有数组和 scanner_cfg 背景信息，因此严格物理对齐不可验证，只能 approximate。
5. 如果正式论文表格要求公平比较，必须明确所有模型最终是否都进入同一个 reference GT grid。如果某些模型只能在自己的 processed GT grid 验证，则应单独成表或标注 approximate。

---

## 15. Codex 修改优先级

请按以下顺序修改代码：

1. 读取 `model_diet.yaml` 和 `eval_global.yaml`。
2. 实现统一 `load_prediction_by_diet()`。
3. 实现 `apply_axis_transform()`。
4. 实现 `inverse_intensity()`，支持当前所有 formula。
5. 实现 `align_prediction_to_gt()`，支持 identity、resize_to_gt_shape、crop_pad_resize、center_crop_or_pad_z。
6. 修改 `prepare_case_for_metric()`，返回 `gt, pred, mask, debug_info`。
7. 修改 `metric()`，确保 MAE/PSNR mask 内，SSIM/LPIPS 按 slice-wise mask 方案。
8. 输出 `per_case_metrics.csv`、`summary_metrics.csv`、`debug_alignment.csv`、`failed_cases.csv`。
9. 增加 `verify_alignment.py`，批量生成三平面对比图。
10. 对每个模型随机抽 2-3 个 case 先跑 debug 图，确认强度和方向后再跑完整 test set。

---

## 16. 最终验收标准

代码完成后，每个 model-case 必须满足：

```text
pred_final.shape == gt.shape == mask.shape
mask 是 bool
debug_alignment.csv 有完整记录
three_plane_compare 图能显示 GT 与各模型三平面对齐结果
per_case_metrics.csv 有每个 case 的四个指标
summary_metrics.csv 有每个模型的 mean/std
failed_cases.csv 记录缺失文件或对齐失败的 case
```

如果某个模型的 `alignment_quality != verified`，不要静默忽略；在 summary 中保留该字段，必要时在主表外单独说明。
