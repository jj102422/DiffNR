# 多模型 CT 重建结果统一评估实现需求文档

> 目标：将 x2ct、perx2ct、DIF、NAF、3DGS、DiffNR、perx2ct+slicefixer+mask 等模型的重建结果整合到一个统一评估流程中，基于同一套 GT、GT mask、对齐策略和指标函数，计算 test 集所有病例的 MAE、LPIPS、PSNR、SSIM，并输出 per-case 结果与所有 test 病例平均值。
>
> 本文档用于交给 Codex 实现代码。实现时请优先保证评估一致性、可追踪性和可复现性，而不是简单把各模型已有 metric 脚本拼在一起。

---

## 1. 总体原则

### 1.1 统一评估目标

实现一个统一评估入口：

```python
metrics = metric(GT, pred, mask, Diet)
```

其中：

- `GT`：当前 case 的 ground truth CT volume。
- `pred`：当前模型当前 case 的重建 CT volume。
- `mask`：当前 case 的 GT mask，只在 mask 内计算主指标。
- `Diet`：模型处理字典，记录该模型的读取方式、归一化方式、reshape/transpose/flip、裁剪/resize、GT 对齐策略、metric 参数等。

最终目标是：

```text
for model in models:
    for case in test_cases:
        load GT
        load pred
        load mask
        根据 Diet 对 pred 与 GT 对齐
        在 GT mask 内计算 MAE、PSNR
        按 slice-wise mask 策略计算 SSIM
        按 mask bbox crop 策略计算 LPIPS
    汇总该模型所有 test cases 的 mean ± std
```

---

## 2. 从 Excel 表格读取模型配置

用户提供的 Excel 文件包含以下字段：

| 字段名 | 含义 |
|---|---|
| 模型 | 模型名称 |
| GT | 该模型训练或原始评估时使用的 GT 来源 |
| 归一化范围 | 训练或输出归一化使用的 CT 范围 |
| reshape | 该模型输出需要做的 reshape / transpose / flip |
| 分辨率 | 模型输出分辨率 |
| 前向投影 | 训练时或方法中使用的 projector |
| 与GT对齐的处理 | 当前已有的人工对齐说明 |

Codex 需要实现一个读取函数：

```python
def load_excel_config(excel_path: str) -> dict:
    """
    读取模型参数配置表.xlsx，返回原始配置字典。
    返回结构：
    {
        "x2ct": {
            "gt_source": "...",
            "norm_range_text": "[0,2500]",
            "reshape_text": "...",
            "resolution_text": "...",
            "projector": "...",
            "align_note": "..."
        },
        ...
    }
    """
```

注意：

1. Excel 里的内容是“人工记录”和“初始处理说明”，不能直接作为唯一真相。
2. 代码应允许用 YAML/JSON 覆盖 Excel 中的字段。
3. 最终评估必须以统一评估配置 `eval_config.yaml` 为准。
4. Excel 作为模型配置的初始来源和实验记录来源。

---

## 3. 建议的项目结构

请实现如下结构：

```text
evaluation/
├── configs/
│   ├── eval_config.yaml
│   └── model_diet.yaml
├── scripts/
│   ├── build_manifest.py
│   ├── verify_alignment.py
│   ├── evaluate_all.py
│   └── summarize_results.py
├── src/
│   ├── __init__.py
│   ├── config_io.py
│   ├── volume_io.py
│   ├── align.py
│   ├── mask_ops.py
│   ├── metrics.py
│   ├── lpips_metric.py
│   ├── report.py
│   └── visualization.py
└── outputs/
    ├── per_case_metrics.csv
    ├── summary_metrics.csv
    ├── debug_alignment.csv
    ├── failed_cases.csv
    └── visual_check/
```

---

## 4. 核心配置文件设计

### 4.1 eval_config.yaml

这是全局评估配置，控制所有模型共用的规则。

```yaml
dataset:
  test_list: "/home/public/CTSpine1K/data/x2ct_result/test.txt"

  # 统一评估 GT。不要每个模型各自找自己的 GT 算最终表。
  eval_gt_root: "/home/public/CTSpine1K/data/ct512"
  eval_gt_type: "h5"
  eval_gt_h5_name: "ct_xray_data.h5"
  eval_gt_h5_key: "ct"

  # GT mask 路径。具体 pattern 根据你的真实文件修改。
  eval_mask_root: "/home/public/CTSpine1K/data/CT_Spine1K_GT_MASK/CT_Spine1K_GT_MASK/mask/case"
  eval_mask_pattern: "{case_id}.*"

canonical:
  # 统一内部数组方向。
  # 所有 GT、pred、mask 进入 metric 前都必须是 [Z, Y, X]
  axis_order: "ZYX"

  # 统一评估强度范围。
  # 建议最终论文表格固定一个范围，不能每个模型用自己的 data_range。
  # 如果原始 GT 实际是 [0, 2500]，则设 2500。
  # 如果最终决定所有模型都映射到 [0, 3000]，则统一设 3000。
  ct_min: 0.0
  ct_max: 2500.0

  # pred 和 GT 进入 PSNR/SSIM/LPIPS 前是否 clip 到统一范围。
  clip_before_metric: true

metric:
  mae:
    # "raw" 表示 MAE 使用反归一化后的 CT 强度。
    # "norm" 表示 MAE 使用 [0,1] 归一化值。
    # 为了结果清楚，可以同时输出 MAE_raw 和 MAE_norm，但最终主表使用 mae_primary。
    mae_primary: "raw"
    output_both_raw_and_norm: true

  psnr:
    # 推荐使用 normalized [0,1] 后的 mask 内 MSE，因此 data_range=1.0。
    use_normalized: true
    data_range: 1.0
    eps: 1.0e-8

  ssim:
    enabled: true
    use_normalized: true
    data_range: 1.0
    view: "axial"
    min_mask_pixels_per_slice: 100
    gaussian_weights: false
    full_map_mask_average: true

  lpips:
    enabled: true
    view: "axial"
    backbone: "alex"
    device: "cuda"
    batch_size: 16
    resize_hw: [256, 256]
    min_mask_pixels_per_slice: 100
    bbox_padding: 8
    mask_outside_bbox: false

runtime:
  num_workers: 0
  strict_shape_assert: true
  allow_missing_cases: false
  save_debug_npy: false
  save_visual_check: true
  visual_check_num_slices: 5

output:
  output_dir: "./outputs"
  per_case_csv: "per_case_metrics.csv"
  summary_csv: "summary_metrics.csv"
  debug_csv: "debug_alignment.csv"
  failed_csv: "failed_cases.csv"
```

---

### 4.2 model_diet.yaml

这是模型级处理字典。Codex 可以先把 Excel 读取结果转成这个 YAML，再由用户手动修正。

关键思想：**每个模型只负责把自己的 pred 转成统一 canonical 空间；metric 函数不应该关心模型名字。**

```yaml
models:
  x2ct:
    enabled: true
    aliases: ["x2ct", "X2CT"]
    pred_root: "/path/to/x2ct/predictions"
    pred_pattern: "{case_id}.nii.gz"

    train_gt_source: "ct_xray_data_128.h5"
    projector: "Plastimatch DRR"

    pred:
      file_type: "nii.gz"
      raw_axis_order: "ZYX"
      squeeze: true
      remove_channel_dim: true

      # Excel 记录：volume[:, ::-1, :, :]
      # 具体实现时要确认这个 flip 是在 batch/channel 维还是空间 Y 维。
      # 推荐不要直接 hard-code 4D 写法，而是写成空间轴操作。
      transpose: null
      flip_axes: ["Y"]

      normalized: true
      norm_min: 0.0
      norm_max: 2500.0
      clip_normalized_to_0_1: true
      denormalize: true

    align_to_gt:
      strategy: "center_crop_or_pad_z"
      source_shape_hint: [512, 512, 512]
      target: "eval_gt"
      resize_xy: false
      interpolation: "linear"

  perx2ct:
    enabled: true
    aliases: ["perx2ct", "PerX2CT"]
    pred_root: "/path/to/perx2ct/predictions"
    pred_pattern: "{case_id}.npy"

    train_gt_source: "ct512_CTSlice_npz"
    projector: "Plastimatch DRR"

    pred:
      file_type: "npy"
      raw_axis_order: "DHW"
      # Excel 记录：[D,H,W] -> [W,H,D]
      # 但统一评估内部要求 ZYX，所以需要实际检查。
      # 如果原始就是 [Z,Y,X]，transpose 应设为 null。
      transpose: null
      flip_axes: []
      normalized: true
      norm_min: 0.0
      norm_max: 2500.0
      clip_normalized_to_0_1: true
      denormalize: true

    align_to_gt:
      strategy: "assert_same_shape"
      target: "eval_gt"
      interpolation: "linear"

  DIF:
    enabled: true
    aliases: ["DIF", "dif"]
    pred_root: "/path/to/dif/predictions"
    pred_pattern: "{case_id}.npy"

    train_gt_source: "ct512"
    projector: "Plastimatch DRR"

    pred:
      file_type: "npy"
      # Excel 记录：pred 可能是 [512,512,D] or flat，reshape 后 transpose 到 [D,512,512]
      raw_axis_order: "XYD_or_flat"
      reshape_to: [512, 512, "{gt_z}"]
      transpose: [2, 1, 0]
      flip_axes: []
      normalized: true
      norm_min: 0.0
      norm_max: 2500.0
      clip_normalized_to_0_1: true
      denormalize: true

    align_to_gt:
      strategy: "assert_same_shape"
      target: "eval_gt"
      interpolation: "linear"

  NAF:
    enabled: true
    aliases: ["NAF", "naf"]
    pred_root: "/path/to/naf/predictions"
    pred_pattern: "{case_id}/image_pred.nii.gz"

    train_gt_source: "ct512"
    projector: "TIGRE"

    pred:
      file_type: "nii.gz"
      # Excel 记录：np.transpose(image_np, (2,1,0)) 转为 z,y,x
      raw_axis_order: "XYZ"
      transpose: [2, 1, 0]
      flip_axes: []
      normalized: true
      norm_min: 0.0
      norm_max: 2500.0
      clip_normalized_to_0_1: true
      denormalize: true

    align_to_gt:
      strategy: "resize_to_gt_shape"
      target: "eval_gt"
      interpolation: "linear"
      preserve_z: true

  3DGuassian:
    enabled: true
    aliases: ["3DGuassian", "3DGaussian", "3DGS", "gs"]
    pred_root: "/path/to/3dgs/predictions"
    pred_pattern: "{case_id}.npy"

    train_gt_source: "ct512_CTSlice_npz"
    projector: "TIGRE"

    pred:
      file_type: "npy"
      raw_axis_order: "XYZ_or_ZYX_need_verify"
      transpose: null
      flip_axes: []
      normalized: true
      norm_min: 0.0
      norm_max: 3000.0
      clip_normalized_to_0_1: true
      denormalize: true

    align_to_gt:
      strategy: "center_crop_or_pad_z"
      source_shape_hint: [512, 512, 512]
      target: "eval_gt"
      interpolation: "linear"

  DiffNR:
    enabled: true
    aliases: ["DiffNR", "diffnr"]
    pred_root: "/path/to/diffnr/predictions"
    pred_pattern: "{case_id}/vol_pred.npy"

    train_gt_source: "ct513_CTSlice_npz"
    projector: "TIGRE"

    pred:
      file_type: "npy"
      raw_axis_order: "XYZ_or_ZYX_need_verify"
      transpose: null
      flip_axes: []
      normalized: true
      norm_min: 0.0
      norm_max: 3000.0
      clip_normalized_to_0_1: true
      denormalize: true

    align_to_gt:
      strategy: "center_crop_or_pad_z"
      source_shape_hint: [512, 512, 512]
      target: "eval_gt"
      interpolation: "linear"

  perx2ct_slicefixer_mask:
    enabled: true
    aliases: ["perx2ct+slicefixer+mask"]
    pred_root: "/path/to/perx2ct_slicefixer_mask/predictions"
    pred_pattern: "{case_id}.npy"

    train_gt_source: "ct514_CTSlice_npz"
    projector: "TIGRE"

    pred:
      file_type: "npy"
      raw_axis_order: "ZYX"
      transpose: null
      flip_axes: []
      normalized: true
      norm_min: 0.0
      norm_max: 2500.0
      clip_normalized_to_0_1: true
      denormalize: true

    align_to_gt:
      strategy: "assert_same_shape"
      target: "eval_gt"
      interpolation: "linear"
```

---

## 5. 数据内部标准

所有 volume 在进入 `metric()` 前必须满足：

```python
GT.shape == pred.shape == mask.shape
GT.dtype == np.float32
pred.dtype == np.float32
mask.dtype == bool
axis_order == "ZYX"
```

其中：

- `Z`：slice/depth 方向。
- `Y`：height/row。
- `X`：width/column。

不要在 metric 函数内部做复杂 reshape。正确流程是：

```python
GT, pred, mask, debug = prepare_case_for_metric(case_id, model_name, Diet)
metrics = metric(GT, pred, mask, Diet)
```

---

## 6. volume 读取函数

需要支持以下格式：

```python
def load_volume(path: str, file_type: str, key: str | None = None) -> np.ndarray:
    """
    支持：
    - .npy
    - .npz
    - .nii / .nii.gz
    - .mha / .mhd
    - .h5 / .hdf5
    - .pt / .pth 可选
    返回 np.ndarray，不在这里做归一化。
    """
```

实现要求：

1. `.h5` 必须通过 key 读取，例如 `ct`。
2. `.npz` 如果有多个 key，优先读取配置中的 key；否则报错。
3. `.nii.gz` 和 `.mha/.mhd` 读取后要明确数组轴顺序。
   - SimpleITK 的 `GetArrayFromImage()` 通常返回 `[Z,Y,X]`。
   - nibabel 读取 NIfTI 后常见为 `[X,Y,Z]`，需要由 `Diet.pred.raw_axis_order` 指定转换。
4. 所有读入数据先转成 `np.float32`，mask 例外。
5. 必须处理 NaN/Inf：

```python
if not np.isfinite(volume).all():
    volume = np.nan_to_num(volume, nan=0.0, posinf=max_value, neginf=min_value)
```

---

## 7. 对齐流程设计

### 7.1 总入口

```python
def prepare_case_for_metric(
    case_id: str,
    model_name: str,
    global_cfg: dict,
    model_diet: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    返回：
    GT_raw_eval: [Z,Y,X], float32
    pred_raw_eval: [Z,Y,X], float32
    mask_bool: [Z,Y,X], bool
    debug_info: dict
    """
```

处理顺序必须固定：

```text
1. load GT
2. load mask
3. load pred
4. squeeze pred 的 batch/channel 维
5. 按 Diet reshape pred
6. 按 Diet transpose pred 到 ZYX
7. 按 Diet flip pred
8. pred 反归一化到 CT 强度
9. GT 和 pred clip 到统一 eval range
10. pred 对齐到 GT shape
11. mask 对齐到 GT shape
12. assert GT/pred/mask shape 完全一致
13. 返回给 metric()
```

---

### 7.2 squeeze 规则

很多模型输出可能带有 batch/channel 维，例如：

```text
[1, 1, Z, Y, X]
[1, Z, Y, X]
[Z, Y, X, 1]
```

实现函数：

```python
def squeeze_volume(arr: np.ndarray, remove_channel_dim: bool = True) -> np.ndarray:
    """
    删除 size=1 的 batch/channel 维，但不能误删 Z=1 的真实维度。
    如果 squeeze 后不是 3D，则根据 Diet 显式 reshape。
    """
```

注意：不要盲目 `np.squeeze()` 后就继续，因为可能把真实 depth=1 的 case 弄没。建议：

1. 如果配置里给了 `expected_ndim`，按配置处理。
2. 如果维度数为 5 且前两维为 1，去掉前两维。
3. 如果维度数为 4 且某一维为 1，根据配置删除 channel 维。
4. 最终不是 3D 就报错或按 `reshape_to` 处理。

---

### 7.3 reshape_to 支持 gt_z

DIF 表格里提到：

```python
pred = pred.reshape(512, 512, gt.shape[0])
pred = np.transpose(pred, (2, 1, 0))
```

所以 `reshape_to` 需要支持占位符：

```yaml
reshape_to: [512, 512, "{gt_z}"]
```

实现时：

```python
shape = []
for item in reshape_to:
    if item == "{gt_z}":
        shape.append(GT.shape[0])
    elif item == "{gt_y}":
        shape.append(GT.shape[1])
    elif item == "{gt_x}":
        shape.append(GT.shape[2])
    else:
        shape.append(int(item))
pred = pred.reshape(shape)
```

如果元素数量不匹配，必须报错并写入 failed_cases.csv。

---

### 7.4 axis transpose

统一内部使用 `[Z,Y,X]`。

通用函数：

```python
def apply_axis_transform(pred: np.ndarray, pred_cfg: dict) -> np.ndarray:
    if pred_cfg.get("transpose") is not None:
        pred = np.transpose(pred, tuple(pred_cfg["transpose"]))

    for axis_name in pred_cfg.get("flip_axes", []):
        axis = {"Z": 0, "Y": 1, "X": 2}[axis_name]
        pred = np.flip(pred, axis=axis)

    return pred
```

所有模型的特殊轴变换都必须写在 `Diet` 里，不要散落在代码 if/else 中。

---

### 7.5 反归一化与统一强度范围

模型输出可能是 `[0,1]`，也可能已经是 CT 强度。

实现：

```python
def restore_intensity(pred: np.ndarray, pred_cfg: dict) -> np.ndarray:
    if pred_cfg.get("normalized", False):
        if pred_cfg.get("clip_normalized_to_0_1", True):
            pred = np.clip(pred, 0.0, 1.0)
        if pred_cfg.get("denormalize", True):
            norm_min = float(pred_cfg["norm_min"])
            norm_max = float(pred_cfg["norm_max"])
            pred = pred * (norm_max - norm_min) + norm_min
    return pred.astype(np.float32)
```

之后在 metric 前做统一 eval clip：

```python
def clip_to_eval_range(gt, pred, ct_min, ct_max):
    gt = np.clip(gt, ct_min, ct_max)
    pred = np.clip(pred, ct_min, ct_max)
    return gt.astype(np.float32), pred.astype(np.float32)
```

重要要求：

- 不允许对每个 pred 单独 min-max。
- 不允许对每个 case 单独用 `pred.min()`、`pred.max()` 归一化。
- 归一化只能用全局固定范围 `ct_min`、`ct_max`。
- `pred.norm_max` 是为了把模型输出还原到 CT 强度，`canonical.ct_max` 是为了统一 metric 的归一化，两者概念不同。

---

### 7.6 pred 对齐到 GT shape

实现：

```python
def align_pred_to_gt(pred, gt, align_cfg):
    strategy = align_cfg["strategy"]

    if strategy == "assert_same_shape":
        assert pred.shape == gt.shape
        return pred

    if strategy == "center_crop_or_pad_z":
        pred = center_crop_or_pad_z(pred, target_z=gt.shape[0])
        if pred.shape[1:] != gt.shape[1:]:
            pred = resize_volume_to_shape(pred, gt.shape, interpolation="linear")
        return pred

    if strategy == "resize_to_gt_shape":
        return resize_volume_to_shape(pred, gt.shape, interpolation="linear")

    raise ValueError(f"Unknown align strategy: {strategy}")
```

#### center_crop_or_pad_z

```python
def center_crop_or_pad_z(vol: np.ndarray, target_z: int) -> np.ndarray:
    z, y, x = vol.shape

    if z == target_z:
        return vol

    if z > target_z:
        start = (z - target_z) // 2
        return vol[start:start + target_z, :, :]

    if z < target_z:
        pad_total = target_z - z
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        return np.pad(
            vol,
            ((pad_before, pad_after), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0
        )
```

注意：Excel 中 3DGS/DiffNR 的说明写的是：

```python
start = (512 - gt.shape[2]) // 2
pred = pred[:, :, start:start + gt.shape[2]]
```

这表示当 pred 是 `[X,Y,Z]` 时裁最后一维。但统一内部转成 `[Z,Y,X]` 后，应该裁第一维：

```python
start = (pred.shape[0] - gt.shape[0]) // 2
pred = pred[start:start + gt.shape[0], :, :]
```

所以必须先完成 axis_order 标准化，再执行 crop/pad。

---

### 7.7 mask 对齐

mask 是 GT mask，因此最好直接从 GT 对应 mask 文件读取并转成 `[Z,Y,X]`。如果 mask shape 与 GT 不一致：

```python
mask = resize_mask_to_shape(mask, gt.shape)
```

mask 重采样必须使用 nearest-neighbor：

```python
def resize_mask_to_shape(mask, target_shape):
    # interpolation = nearest
    # 输出 bool
```

不要对 mask 使用 linear interpolation。

---

## 8. metric(GT, pred, mask, Diet) 设计

### 8.1 函数签名

```python
def metric(
    GT: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    Diet: dict
) -> dict:
    """
    输入：
        GT: [Z,Y,X], float32，已 clip 到统一 CT range
        pred: [Z,Y,X], float32，已对齐到 GT
        mask: [Z,Y,X], bool，GT mask
        Diet: 包含 canonical 和 metric 配置

    输出：
        {
            "MAE": float,
            "MAE_raw": float,
            "MAE_norm": float,
            "PSNR": float,
            "SSIM": float,
            "LPIPS": float,
            "mask_voxels": int,
            "valid_ssim_slices": int,
            "valid_lpips_slices": int
        }
    """
```

### 8.2 输入检查

```python
assert GT.ndim == 3
assert pred.ndim == 3
assert mask.ndim == 3
assert GT.shape == pred.shape == mask.shape
assert mask.dtype == bool or mask.dtype == np.bool_
assert np.isfinite(GT).all()
assert np.isfinite(pred).all()
assert mask.sum() > 0
```

如果 `mask.sum() == 0`，该 case 应失败并记录到 `failed_cases.csv`，不要返回假指标。

---

## 9. 归一化工具

```python
def normalize_for_metric(volume, ct_min, ct_max):
    volume = np.clip(volume, ct_min, ct_max)
    return ((volume - ct_min) / (ct_max - ct_min)).astype(np.float32)
```

在 `metric()` 内：

```python
ct_min = Diet["canonical"]["ct_min"]
ct_max = Diet["canonical"]["ct_max"]

GT_norm = normalize_for_metric(GT, ct_min, ct_max)
pred_norm = normalize_for_metric(pred, ct_min, ct_max)
```

---

## 10. MAE 计算

### 10.1 mask 内 raw MAE

```python
mae_raw = np.mean(np.abs(pred[mask] - GT[mask]))
```

### 10.2 mask 内 normalized MAE

```python
mae_norm = np.mean(np.abs(pred_norm[mask] - GT_norm[mask]))
```

### 10.3 主 MAE

```python
if mae_primary == "raw":
    MAE = mae_raw
else:
    MAE = mae_norm
```

建议输出两列：

```text
MAE
MAE_raw
MAE_norm
```

如果论文主表只要一个 MAE，使用 `MAE`。

---

## 11. PSNR 计算

要求：PSNR 在 mask 内计算，使用 normalized `[0,1]` 的 MSE，`data_range=1.0`。

```python
def compute_mask_psnr(gt_norm, pred_norm, mask, eps=1e-8):
    diff = pred_norm[mask] - gt_norm[mask]
    mse = np.mean(diff ** 2)
    if mse <= eps:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)
```

如果你希望用 raw CT 值计算，也可以：

```python
mse_raw = np.mean((pred[mask] - GT[mask]) ** 2)
psnr_raw = 10 * np.log10(((ct_max - ct_min) ** 2) / mse_raw)
```

但最终主表建议统一使用 normalized PSNR，避免不同 CT max 混乱。

---

## 12. SSIM 计算

SSIM 不是 voxel-wise 指标，不能简单写：

```python
structural_similarity(GT[mask], pred[mask])
```

推荐方案：**slice-wise SSIM map + mask 内平均**。

### 12.1 axial slice 计算

```python
from skimage.metrics import structural_similarity

def compute_masked_ssim_slice_map(
    gt_norm: np.ndarray,
    pred_norm: np.ndarray,
    mask: np.ndarray,
    data_range: float = 1.0,
    min_mask_pixels_per_slice: int = 100
) -> tuple[float, int]:

    scores = []

    Z = gt_norm.shape[0]
    for z in range(Z):
        m = mask[z]
        if int(m.sum()) < min_mask_pixels_per_slice:
            continue

        gt_slice = gt_norm[z]
        pred_slice = pred_norm[z]

        # full=True 返回 ssim_map
        _, ssim_map = structural_similarity(
            gt_slice,
            pred_slice,
            data_range=data_range,
            full=True
        )

        scores.append(float(np.mean(ssim_map[m])))

    if len(scores) == 0:
        return float("nan"), 0

    return float(np.mean(scores)), len(scores)
```

### 12.2 为什么不把 mask 外置 0 后算整图 SSIM

不建议：

```python
gt_slice[~mask] = 0
pred_slice[~mask] = 0
ssim = structural_similarity(gt_slice, pred_slice)
```

原因：

1. 背景区域很大时，SSIM 会被大量相同背景拉高。
2. mask 外置 0 会引入人工边界。
3. 不同病例 mask 大小不同，背景比例不同，会影响公平性。

推荐使用 full SSIM map 后只平均 mask 内区域。

---

## 13. LPIPS 计算

LPIPS 是 2D perceptual metric。3D CT 需要转成 2D slices 计算。

推荐策略：**axial slice + mask bounding box crop + resize + 灰度复制成 3 通道 + [-1,1] 输入**。

### 13.1 LPIPS 模型初始化

```python
import lpips
import torch

class LPIPSMetric:
    def __init__(self, net="alex", device="cuda", batch_size=16):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = lpips.LPIPS(net=net).to(self.device)
        self.model.eval()
        self.batch_size = batch_size
```

### 13.2 bbox crop

```python
def get_2d_bbox(mask2d: np.ndarray, padding: int = 8):
    ys, xs = np.where(mask2d)
    if len(ys) == 0:
        return None

    y1 = max(int(ys.min()) - padding, 0)
    y2 = min(int(ys.max()) + padding + 1, mask2d.shape[0])
    x1 = max(int(xs.min()) - padding, 0)
    x2 = min(int(xs.max()) + padding + 1, mask2d.shape[1])
    return y1, y2, x1, x2
```

### 13.3 单 slice 预处理

```python
def prepare_lpips_slice(img2d_norm: np.ndarray, bbox, resize_hw=(256, 256)):
    """
    img2d_norm: [H,W], range [0,1]
    返回 torch tensor [3,H,W], range [-1,1]
    """
    y1, y2, x1, x2 = bbox
    crop = img2d_norm[y1:y2, x1:x2]

    # resize 到固定大小
    crop_t = torch.from_numpy(crop).float()[None, None, :, :]  # [1,1,H,W]
    crop_t = torch.nn.functional.interpolate(
        crop_t,
        size=resize_hw,
        mode="bilinear",
        align_corners=False
    )

    crop_t = crop_t.repeat(1, 3, 1, 1)  # [1,3,H,W]
    crop_t = crop_t * 2.0 - 1.0
    return crop_t[0]
```

### 13.4 整个 volume 的 LPIPS

```python
def compute_lpips_volume(
    gt_norm,
    pred_norm,
    mask,
    lpips_runner,
    resize_hw=(256, 256),
    min_mask_pixels_per_slice=100,
    bbox_padding=8
):
    gt_batch = []
    pred_batch = []
    scores = []
    valid_slices = 0

    Z = gt_norm.shape[0]
    for z in range(Z):
        m = mask[z]
        if int(m.sum()) < min_mask_pixels_per_slice:
            continue

        bbox = get_2d_bbox(m, padding=bbox_padding)
        if bbox is None:
            continue

        gt_t = prepare_lpips_slice(gt_norm[z], bbox, resize_hw=resize_hw)
        pred_t = prepare_lpips_slice(pred_norm[z], bbox, resize_hw=resize_hw)

        gt_batch.append(gt_t)
        pred_batch.append(pred_t)
        valid_slices += 1

        if len(gt_batch) == lpips_runner.batch_size:
            scores.extend(lpips_runner.forward_batch(gt_batch, pred_batch))
            gt_batch.clear()
            pred_batch.clear()

    if len(gt_batch) > 0:
        scores.extend(lpips_runner.forward_batch(gt_batch, pred_batch))

    if len(scores) == 0:
        return float("nan"), 0

    return float(np.mean(scores)), valid_slices
```

```python
class LPIPSMetric:
    ...
    @torch.no_grad()
    def forward_batch(self, gt_list, pred_list):
        gt = torch.stack(gt_list, dim=0).to(self.device)
        pred = torch.stack(pred_list, dim=0).to(self.device)
        dist = self.model(gt, pred)
        return dist.detach().flatten().cpu().numpy().astype(float).tolist()
```

注意：

1. LPIPS 越低越好。
2. 输入必须是 3 通道，所以 CT 灰度 slice 需要复制成 3 通道。
3. 输入范围必须是 `[-1,1]`。
4. 不建议整张 512x512 slice 直接算，因为大量背景会降低区分度。
5. 不建议只取 mask 像素形成不规则图像，因为 LPIPS 需要规则 2D 图像。
6. 默认只算 axial view；如果以后想更严格，可以扩展为 axial/coronal/sagittal 三视图平均。

---

## 14. metric() 完整伪代码

```python
def metric(GT, pred, mask, Diet):
    # 1. assert
    assert GT.ndim == pred.ndim == mask.ndim == 3
    assert GT.shape == pred.shape == mask.shape
    mask = mask.astype(bool)
    if mask.sum() == 0:
        raise ValueError("Empty mask")

    ct_min = float(Diet["canonical"]["ct_min"])
    ct_max = float(Diet["canonical"]["ct_max"])

    # 2. clip
    GT = np.clip(GT.astype(np.float32), ct_min, ct_max)
    pred = np.clip(pred.astype(np.float32), ct_min, ct_max)

    # 3. normalized for PSNR/SSIM/LPIPS
    GT_norm = (GT - ct_min) / (ct_max - ct_min)
    pred_norm = (pred - ct_min) / (ct_max - ct_min)
    GT_norm = np.clip(GT_norm, 0.0, 1.0)
    pred_norm = np.clip(pred_norm, 0.0, 1.0)

    # 4. MAE
    mae_raw = float(np.mean(np.abs(pred[mask] - GT[mask])))
    mae_norm = float(np.mean(np.abs(pred_norm[mask] - GT_norm[mask])))

    mae_primary = Diet["metric"]["mae"].get("mae_primary", "raw")
    MAE = mae_raw if mae_primary == "raw" else mae_norm

    # 5. PSNR
    diff = pred_norm[mask] - GT_norm[mask]
    mse = float(np.mean(diff ** 2))
    eps = float(Diet["metric"]["psnr"].get("eps", 1e-8))
    PSNR = float("inf") if mse <= eps else float(10.0 * np.log10(1.0 / mse))

    # 6. SSIM
    SSIM, valid_ssim_slices = compute_masked_ssim_slice_map(
        GT_norm,
        pred_norm,
        mask,
        data_range=1.0,
        min_mask_pixels_per_slice=Diet["metric"]["ssim"]["min_mask_pixels_per_slice"]
    )

    # 7. LPIPS
    LPIPS, valid_lpips_slices = compute_lpips_volume(
        GT_norm,
        pred_norm,
        mask,
        lpips_runner=Diet["_runtime"]["lpips_runner"],
        resize_hw=tuple(Diet["metric"]["lpips"]["resize_hw"]),
        min_mask_pixels_per_slice=Diet["metric"]["lpips"]["min_mask_pixels_per_slice"],
        bbox_padding=Diet["metric"]["lpips"]["bbox_padding"]
    )

    return {
        "MAE": MAE,
        "MAE_raw": mae_raw,
        "MAE_norm": mae_norm,
        "PSNR": PSNR,
        "SSIM": SSIM,
        "LPIPS": LPIPS,
        "mask_voxels": int(mask.sum()),
        "mse_norm_mask": mse,
        "valid_ssim_slices": int(valid_ssim_slices),
        "valid_lpips_slices": int(valid_lpips_slices),
    }
```

---

## 15. evaluate_all.py 设计

### 15.1 命令行

```bash
python scripts/evaluate_all.py \
  --eval_config configs/eval_config.yaml \
  --model_diet configs/model_diet.yaml \
  --excel_config /mnt/data/模型参数配置表.xlsx \
  --models x2ct perx2ct DIF NAF 3DGuassian DiffNR perx2ct_slicefixer_mask \
  --output_dir outputs
```

### 15.2 主流程

```python
def main():
    global_cfg = load_yaml(args.eval_config)
    model_diet = load_yaml(args.model_diet)

    if args.excel_config:
        excel_cfg = load_excel_config(args.excel_config)
        model_diet = merge_excel_config(model_diet, excel_cfg)

    test_cases = read_test_list(global_cfg["dataset"]["test_list"])

    lpips_runner = LPIPSMetric(
        net=global_cfg["metric"]["lpips"]["backbone"],
        device=global_cfg["metric"]["lpips"]["device"],
        batch_size=global_cfg["metric"]["lpips"]["batch_size"],
    )

    all_rows = []
    debug_rows = []
    failed_rows = []

    for model_name in args.models:
        cfg = model_diet["models"][model_name]
        if not cfg.get("enabled", True):
            continue

        Diet = build_runtime_diet(global_cfg, cfg)
        Diet["_runtime"] = {"lpips_runner": lpips_runner}

        for case_id in test_cases:
            try:
                GT, pred, mask, debug = prepare_case_for_metric(
                    case_id=case_id,
                    model_name=model_name,
                    global_cfg=global_cfg,
                    model_diet=cfg
                )

                result = metric(GT, pred, mask, Diet)
                row = {
                    "model": model_name,
                    "case_id": case_id,
                    **result
                }
                all_rows.append(row)

                debug_rows.append({
                    "model": model_name,
                    "case_id": case_id,
                    **debug
                })

            except Exception as e:
                failed_rows.append({
                    "model": model_name,
                    "case_id": case_id,
                    "error": repr(e)
                })
                if not global_cfg["runtime"].get("allow_missing_cases", False):
                    raise

    save_csv(all_rows, output_dir / "per_case_metrics.csv")
    save_csv(debug_rows, output_dir / "debug_alignment.csv")
    save_csv(failed_rows, output_dir / "failed_cases.csv")

    summary = summarize_metrics(all_rows)
    save_csv(summary, output_dir / "summary_metrics.csv")
```

---

## 16. 输出 CSV 设计

### 16.1 per_case_metrics.csv

列名建议：

```text
model
case_id
MAE
MAE_raw
MAE_norm
PSNR
SSIM
LPIPS
mask_voxels
mse_norm_mask
valid_ssim_slices
valid_lpips_slices
```

### 16.2 summary_metrics.csv

每个模型一行：

```text
model
n_cases
MAE_mean
MAE_std
LPIPS_mean
LPIPS_std
PSNR_mean
PSNR_std
SSIM_mean
SSIM_std
MAE_raw_mean
MAE_raw_std
MAE_norm_mean
MAE_norm_std
```

汇总函数：

```python
def summarize_metrics(rows):
    df = pd.DataFrame(rows)
    metric_names = ["MAE", "LPIPS", "PSNR", "SSIM", "MAE_raw", "MAE_norm"]
    summary_rows = []

    for model, g in df.groupby("model"):
        row = {"model": model, "n_cases": len(g)}
        for m in metric_names:
            vals = pd.to_numeric(g[m], errors="coerce")
            row[f"{m}_mean"] = vals.mean()
            row[f"{m}_std"] = vals.std(ddof=1)
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)
```

注意：

- 对每个 case 先算指标，再对 case 平均。
- 不要把所有 case 的 voxels 拼起来算一个整体 MAE。
- case-level mean 更公平，因为每个病例权重相同。
- 如果某个模型缺 case，必须在 summary 里写 `n_cases`，不能悄悄忽略。

---

## 17. debug_alignment.csv 设计

每个 case 记录：

```text
model
case_id
gt_path
pred_path
mask_path
gt_shape_raw
pred_shape_raw
mask_shape_raw
pred_shape_after_squeeze
pred_shape_after_axis
pred_shape_after_intensity
pred_shape_after_align
mask_shape_after_align
gt_min
gt_max
pred_min_raw
pred_max_raw
pred_min_after_intensity
pred_max_after_intensity
mask_voxels
align_strategy
transpose
flip_axes
norm_min
norm_max
canonical_ct_min
canonical_ct_max
```

这个文件非常重要。它可以帮你快速发现：

1. 某个模型方向错了。
2. 某个模型强度范围没反归一化。
3. 某个模型 Z 维裁剪错了。
4. 某个 mask 没对齐。
5. 某个 case 不是同一个 GT。

---

## 18. 视觉检查图

建议每个模型随机或固定保存若干 case 的对齐图：

```text
outputs/visual_check/
├── x2ct/
│   ├── case0051_axial.png
│   ├── case0051_sagittal.png
│   └── case0051_coronal.png
├── DiffNR/
...
```

每张图包含：

```text
GT | pred | abs error | mask overlay
```

至少取：

1. mask voxel 数最多的 axial slice。
2. mask bbox 中心 slice。
3. 随机 3 张有 mask 的 slice。

实现：

```python
def save_visual_check(gt, pred, mask, out_path):
    """
    保存 2D 对比图。
    不要求美观，但要能快速确认方向、裁剪、强度是否合理。
    """
```

在正式跑全部指标前，必须先对每个模型至少抽 1 个 case 做视觉检查。

---

## 19. 模型特殊处理说明

以下内容来自用户 Excel 表格，应转成 `Diet` 后实现。

### 19.1 x2ct

Excel 信息：

```text
GT: ct_xray_data_128.h5
归一化范围: [0,2500]
reshape: volume[:, ::-1, :, :]
分辨率: [128,128,128]
前向投影: Plastimatch DRR
```

已有对齐说明中出现：

```python
gt = load_gt()          # 512 x 512 x Z
pred = load_pred()      # 512 x 512 x 512

start = (512 - gt.shape[2]) // 2
pred = pred[:, :, start:start + gt.shape[2]]

pred = np.clip(pred, 0, 1)

assert pred.shape == gt.shape
```

实现注意：

1. 如果 pred 进入内部后是 `[Z,Y,X]`，中心裁剪应该裁 `axis=0`，不是最后一维。
2. `volume[:, ::-1, :, :]` 需要确认是空间 Y flip 还是 batch/channel 维相关操作。
3. 如果 x2ct pred 已经是 `[0,1]`，先 clip，再乘 2500 反归一化。
4. 不要使用 `ct_xray_data_128.h5` 作为最终统一评估 GT，除非全局 eval_config 明确选择它。建议最终统一用 `ct512/<case>/ct_xray_data.h5`。

---

### 19.2 perx2ct

Excel 信息：

```text
GT: ct512_CTSlice_npz
归一化范围: [0,2500]
reshape: [D,H,W] -> [W,H,D]
分辨率: [512,512,D]
前向投影: Plastimatch DRR
```

已有说明：

```text
1. load original GT
2. load PerX2CT pred
3. check axis order
4. assert pred.shape == gt.shape
5. do not crop Z
6. do not modify GT
7. pred = clip(pred, 0, 1)
8. pred = pred * 2500
```

实现注意：

1. perx2ct 通常不裁 Z。
2. 如果 pred 已经和 GT shape 一致，`align_to_gt.strategy = assert_same_shape`。
3. 表格中的 `[D,H,W] -> [W,H,D]` 和统一内部 `[Z,Y,X]` 可能冲突，需要用视觉检查确认。
4. 代码允许配置 `transpose`，不要写死。

---

### 19.3 DIF

Excel 信息：

```text
GT: ct512
归一化范围: [0,2500]
reshape: /
分辨率: [512,513,D]
前向投影: Plastimatch DRR
```

已有说明：

```python
gt = load_original_gt(gt_path).astype(np.float32)   # [D, 512, 512]

pred = load_dif_pred(pred_path).astype(np.float32)  # [512, 512, D] or flat
pred = pred.reshape(512, 512, gt.shape[0])
pred = np.transpose(pred, (2, 1, 0))                # -> [D, 512, 512]

assert pred.shape == gt.shape

pred = np.clip(pred, 0, 1)
pred = pred * 2500.0
```

实现注意：

1. DIF 可能是 flat array，需要支持 `reshape_to`。
2. `transpose: [2,1,0]` 应配置在 Diet。
3. 若实际分辨率里出现 513，不能静默 resize，必须 debug 记录并确认是否需要 crop 到 512。
4. 如果 513 是 detector 或中间表示，不一定是最终 pred shape。

---

### 19.4 NAF

Excel 信息：

```text
GT: ct512
归一化范围: [0,2500]
reshape: np.transpose(image_np, (2,1,0)) 转为 z,y,x
分辨率: [256,256,D]
前向投影: TIGRE
```

已有说明：

```text
1. load original gt: ct_file.mha
2. load pred: image_pred.nii.gz
3. pred clip 到 [0,1]
4. pred 反归一化: pred = pred * 2500
5. pred resize 到 gt.shape，不裁 Z
6. assert pred.shape == gt.shape
```

实现注意：

1. NAF 输出可能是 `[256,256,D]`，需要 resize 到 GT `[Z,512,512]`。
2. pred 用 linear interpolation。
3. mask 用 nearest interpolation。
4. 不裁 Z，除非实际 pred Z 与 GT Z 不一致且用户在 Diet 中指定。
5. NIfTI 读取轴顺序需要特别检查。

---

### 19.5 3DGuassian / 3DGaussian / 3DGS

Excel 信息：

```text
GT: ct512_CTSlice_npz
归一化范围: [0,3000]
reshape: /
分辨率: [512,512,512]
前向投影: TIGRE
```

已有说明：

```text
1. 先把 pred 从 512³ 对齐到 GT 的 512x512xZ。
2. Z 维不能直接拿前 Z 张，应该按生成 512³ volume_gt 的逻辑反向裁掉 padding。
3. start = (512 - gt.shape[2]) // 2
4. pred = pred[:, :, start:start+gt.shape[2]]
5. pred = clip(pred, 0, 1)
```

实现注意：

1. 模型名表格中写作 `3DGuassian`，但常用写法是 `3DGaussian`，代码要支持 alias。
2. 先转成 `[Z,Y,X]` 后再中心裁剪 Z。
3. pred 输出 `[0,1]` 时乘 3000 反归一化。
4. 最终 metric 使用统一 `canonical.ct_max`，不要因为这个模型是 3000 就单独用 3000 算 PSNR/SSIM，除非全局配置也设为 3000。

---

### 19.6 DiffNR

Excel 信息：

```text
GT: ct513_CTSlice_npz
归一化范围: [0,3000]
reshape: /
分辨率: [512,512,512]
前向投影: TIGRE
```

已有说明：

```text
1. 中心裁 Z 到原始 GT shape。
2. pred clip 到 [0,1]。
3. 再和 GT 比。
```

实现注意：

1. 和 3DGS 类似，先标准化 axis，再中心裁 Z。
2. pred 乘 3000 反归一化。
3. DiffNR 的 `volume_gt.npy` 可能已经是 padded/cropped 版本，最终统一评估仍建议回到 eval_gt。
4. 如果 DiffNR 的 GT 和 eval_gt 不是同一个空间，必须明确记录，否则不能直接与其他模型公平比较。

---

### 19.7 perx2ct+slicefixer+mask

Excel 信息：

```text
GT: ct514_CTSlice_npz
归一化范围: [0,2500]
reshape: /
分辨率: [512,512,D]
前向投影: TIGRE
```

已有说明：

```text
1. shape 对齐检查。
2. 不裁 Z。
3. clip 到 [0,1]。
4. 和原始 GT 比。
```

实现注意：

1. 如果 pred shape 与 GT 一致，直接 assert。
2. pred `[0,1]` 乘 2500。
3. 使用同一 GT mask。
4. 该模型和 perx2ct 的差异应体现在模型结果本身，不应体现在 metric 处理方式上。

---

## 20. 前向投影器公平性说明

表格中记录：

```text
x2ct / perx2ct / DIF: Plastimatch
NAF / 3DGS / DiffNR: TIGRE
```

这些 projector 是训练或重建流程中的差异。对于当前四个 CT volume 指标：

```text
MAE / PSNR / SSIM / LPIPS
```

主评估不需要前向投影器。主评估应该是：

```text
pred CT volume vs GT CT volume
```

而不是：

```text
project(pred CT) vs X-ray
```

因此：

1. `projector` 字段只作为实验记录写入 debug CSV。
2. 不要在 MAE、PSNR、SSIM、LPIPS 中调用 Plastimatch 或 TIGRE。
3. 如果以后要加 projection-domain metric，应单独实现：
   - 所有模型统一使用一个 projector。
   - 同一个 geometry。
   - 同一个 detector size。
   - 同一个 SAD/SDD。
   - 同一个角度。
   - 同一个归一化方式。
4. projection metric 不能和 volume metric 混在一个主表里。

---

## 21. 测试与验证要求

### 21.1 单元测试

请实现以下测试：

#### test_metric_identity

```python
GT = random volume
pred = GT.copy()
mask = random bool mask
```

期望：

```text
MAE = 0
PSNR = inf 或非常大
SSIM 接近 1
LPIPS 接近 0
```

#### test_mask_mae_psnr

构造一个小数组，手算 mask 内 MAE 和 PSNR，确认代码只使用 mask 区域。

#### test_center_crop_z

```python
pred.shape = [512,512,512]
gt.shape = [482,512,512]
```

转成 `[Z,Y,X]` 后应输出：

```python
pred_cropped.shape == gt.shape
start == 15
```

#### test_empty_mask

空 mask 必须报错。

#### test_shape_mismatch

如果对齐后 shape 不一致，必须报错并记录失败。

---

### 21.2 对齐验证

运行全部模型前，先执行：

```bash
python scripts/verify_alignment.py \
  --eval_config configs/eval_config.yaml \
  --model_diet configs/model_diet.yaml \
  --case_id 0051 \
  --models x2ct perx2ct DIF NAF 3DGuassian DiffNR
```

输出：

1. 每个模型的 shape before/after。
2. min/max。
3. visual_check 图片。
4. mask overlay 图片。

只有视觉检查确认方向正确后，才跑全 test。

---

## 22. 错误处理

### 22.1 failed_cases.csv

任何失败都写入：

```text
model
case_id
stage
error
gt_path
pred_path
mask_path
```

`stage` 可以是：

```text
load_gt
load_pred
load_mask
reshape
transpose
intensity
align
metric
save
```

### 22.2 allow_missing_cases

如果：

```yaml
allow_missing_cases: false
```

遇到一个失败就停止。

如果：

```yaml
allow_missing_cases: true
```

记录失败并继续，但 summary 里必须显示实际 `n_cases`。

---

## 23. 最终报告表格

输出 `summary_metrics.csv` 后，建议同时打印 Markdown 表：

```text
| Model | N | MAE ↓ | LPIPS ↓ | PSNR ↑ | SSIM ↑ |
|---|---:|---:|---:|---:|---:|
| x2ct | 20 | mean ± std | mean ± std | mean ± std | mean ± std |
| perx2ct | 20 | mean ± std | mean ± std | mean ± std | mean ± std |
| DIF | 20 | mean ± std | mean ± std | mean ± std | mean ± std |
| NAF | 20 | mean ± std | mean ± std | mean ± std | mean ± std |
| 3DGS | 20 | mean ± std | mean ± std | mean ± std | mean ± std |
| DiffNR | 20 | mean ± std | mean ± std | mean ± std | mean ± std |
```

排序规则建议：

```text
先按用户传入 models 顺序；
不要自动按某个指标排序；
否则论文表格顺序可能不稳定。
```

---

## 24. 实现优先级

### 第一阶段：先把流程跑通

1. 读取 test list。
2. 读取 GT。
3. 读取 mask。
4. 读取一个模型 pred。
5. 完成对齐。
6. 计算 MAE / PSNR。
7. 输出 per_case_metrics.csv。

### 第二阶段：加入 SSIM

1. slice-wise SSIM map。
2. mask 内平均。
3. 记录 valid_ssim_slices。

### 第三阶段：加入 LPIPS

1. 初始化 lpips model。
2. mask bbox crop。
3. slice batch 计算。
4. 记录 valid_lpips_slices。

### 第四阶段：批量模型和可视化

1. 支持所有模型。
2. 输出 debug_alignment.csv。
3. 输出 summary_metrics.csv。
4. 输出 visual_check。

---

## 25. 最重要的禁止事项

实现时不要做以下事情：

1. 不要每个模型用自己的 GT 算最终主表。
2. 不要每个模型用自己的 PSNR data_range 算最终主表。
3. 不要对 pred 使用 case-wise min-max。
4. 不要对 GT 使用 case-wise min-max。
5. 不要把 mask 外背景计入 MAE / PSNR。
6. 不要直接对 `GT[mask]` 和 `pred[mask]` 算 SSIM。
7. 不要对整张 slice 算 LPIPS 后声称是 mask 内 LPIPS。
8. 不要用 linear interpolation resize mask。
9. 不要在 metric 函数内部写模型名 if/else。
10. 不要忽略 axis order 和 flip 的视觉检查。
11. 不要把 projection-domain metric 和 volume-domain metric 混在一起。
12. 不要静默跳过失败 case。

---

## 26. Codex 实现检查清单

实现完成后，请逐项检查：

```text
[ ] 可以读取 Excel 表格。
[ ] 可以读取 eval_config.yaml。
[ ] 可以读取 model_diet.yaml。
[ ] 可以读取 test.txt。
[ ] 每个模型可以通过 Diet 找到 pred path。
[ ] GT/pred/mask 进入 metric 前都是 [Z,Y,X]。
[ ] GT/pred/mask shape 完全一致。
[ ] pred 已经从模型归一化空间还原到 CT 强度。
[ ] GT 和 pred 使用统一 canonical.ct_min / canonical.ct_max clip。
[ ] MAE 在 mask 内计算。
[ ] PSNR 在 mask 内计算。
[ ] SSIM 使用 full ssim_map 后在 mask 内平均。
[ ] LPIPS 使用 mask bbox crop 后逐 slice 计算。
[ ] 输出 per_case_metrics.csv。
[ ] 输出 summary_metrics.csv。
[ ] 输出 debug_alignment.csv。
[ ] 输出 failed_cases.csv。
[ ] 每个模型至少保存一个 visual_check 图。
[ ] pred==GT 的单元测试通过。
[ ] 空 mask 会报错。
[ ] shape mismatch 会报错。
```

---

## 27. 推荐的最终入口代码骨架

```python
# scripts/evaluate_all.py

def main():
    args = parse_args()

    global_cfg = load_yaml(args.eval_config)
    model_diet_all = load_yaml(args.model_diet)

    test_cases = read_test_list(global_cfg["dataset"]["test_list"])

    lpips_runner = None
    if global_cfg["metric"]["lpips"].get("enabled", True):
        lpips_runner = LPIPSMetric(
            net=global_cfg["metric"]["lpips"].get("backbone", "alex"),
            device=global_cfg["metric"]["lpips"].get("device", "cuda"),
            batch_size=global_cfg["metric"]["lpips"].get("batch_size", 16),
        )

    rows = []
    debug_rows = []
    failed_rows = []

    for model_name in args.models:
        model_cfg = model_diet_all["models"][model_name]
        Diet = {
            "canonical": global_cfg["canonical"],
            "metric": global_cfg["metric"],
            "model": model_cfg,
            "_runtime": {
                "lpips_runner": lpips_runner
            }
        }

        for case_id in test_cases:
            try:
                gt, pred, mask, debug = prepare_case_for_metric(
                    case_id=case_id,
                    model_name=model_name,
                    global_cfg=global_cfg,
                    model_diet=model_cfg
                )

                out = metric(gt, pred, mask, Diet)
                rows.append({
                    "model": model_name,
                    "case_id": case_id,
                    **out
                })
                debug_rows.append({
                    "model": model_name,
                    "case_id": case_id,
                    **debug
                })

            except Exception as e:
                failed_rows.append({
                    "model": model_name,
                    "case_id": case_id,
                    "error": repr(e)
                })
                if not global_cfg["runtime"].get("allow_missing_cases", False):
                    raise

    save_per_case(rows, global_cfg)
    save_debug(debug_rows, global_cfg)
    save_failed(failed_rows, global_cfg)
    save_summary(rows, global_cfg)
```

---

## 28. 交付结果

Codex 最终应该交付：

```text
evaluation/
├── configs/
│   ├── eval_config.yaml
│   └── model_diet.yaml
├── scripts/
│   ├── evaluate_all.py
│   ├── verify_alignment.py
│   └── summarize_results.py
├── src/
│   ├── config_io.py
│   ├── volume_io.py
│   ├── align.py
│   ├── mask_ops.py
│   ├── metrics.py
│   ├── lpips_metric.py
│   ├── report.py
│   └── visualization.py
└── tests/
    ├── test_metrics.py
    ├── test_align.py
    └── test_io.py
```

最终运行命令：

```bash
python scripts/evaluate_all.py \
  --eval_config configs/eval_config.yaml \
  --model_diet configs/model_diet.yaml \
  --models x2ct perx2ct DIF NAF 3DGuassian DiffNR perx2ct_slicefixer_mask
```

最终输出：

```text
outputs/per_case_metrics.csv
outputs/summary_metrics.csv
outputs/debug_alignment.csv
outputs/failed_cases.csv
outputs/visual_check/
```

---

## 29. 一句话总结

这个评估系统的核心是：

```text
模型差异全部写进 Diet；
metric 函数只接收已经对齐好的 GT、pred、mask；
所有模型使用同一个 GT 空间、同一个 mask、同一个强度范围、同一套指标实现；
最终对 test cases 做 case-level mean ± std。
```
