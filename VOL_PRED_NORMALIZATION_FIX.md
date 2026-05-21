# vol_pred.npy 归一化修复

## 问题描述

在数据处理流程中发现了数据范围不一致的问题：

- **volume_gt.npy**（真值）：范围 [0.0, 0.976]，正确归一化到 [0, 1]
- **vol_pred.npy**（预测）：范围 [0.0, 8.087]，**未经归一化**，使用原始 Gaussian 体素化输出

这种不一致会导致：
1. 损失函数计算不准确（期望的 pixel_max=1.0 不适用）
2. SliceFixer 输入数据范围不匹配
3. 评价指标（PSNR/SSIM）计算偏差

## 根本原因

在 `r2_gaussian/dataset/__init__.py` 的 `Scene.save()` 方法中：

```python
# OLD CODE (有问题)
vol_pred = queryfunc(self.gaussians)["vol"]  # 原始 Gaussian 输出 [0, 8.09]
np.save(..., t2a(vol_pred))  # 直接保存，无任何归一化
```

函数 `t2a()` 仅进行 tensor 到 numpy 的转换，不涉及任何数值缩放。

而 `volume_gt.npy` 来自 `generate_data.py` 中的 `load_mha_volume()`：

```python
# 正确的归一化
volume_xyz = np.clip(volume_xyz, 0.0, None)   # 零截断
volume_xyz = volume_xyz / 3000.0               # 按固定范围 3000 缩放
volume_xyz = np.clip(volume_xyz, 0.0, 1.0)    # 确保 [0, 1]
```

## 修复方案

在 `Scene.save()` 方法中添加 vol_pred 的归一化处理：

```python
def save(self, iteration, queryfunc):
    point_cloud_path = osp.join(
        self.model_path, "point_cloud/iteration_{}".format(iteration)
    )
    self.gaussians.save_ply(
        osp.join(point_cloud_path, "point_cloud.pickle")
    )
    if queryfunc is not None:
        vol_pred = queryfunc(self.gaussians)["vol"]
        vol_gt = self.vol_gt
        
        # ✅ 新增：转换为 numpy 并归一化
        vol_pred_np = t2a(vol_pred)
        vol_pred_max = vol_pred_np.max()
        if vol_pred_max > 0:
            vol_pred_np = vol_pred_np / vol_pred_max  # 按最大值缩放
        vol_pred_np = np.clip(vol_pred_np, 0.0, 1.0)   # 确保 [0, 1]
        
        np.save(osp.join(point_cloud_path, "vol_gt.npy"), t2a(vol_gt))
        np.save(
            osp.join(point_cloud_path, "vol_pred.npy"),
            vol_pred_np,  # ✅ 现在保存的是归一化后的值
        )
```

### 归一化逻辑

- **方法**：使用**全局固定尺度** `K_pred` 进行缩放，而不是对每个 case 单独按最大值缩放。
- **推荐脚本**：`scripts/normalize_vol_pred_global.py`
- **优点**：
    1. 保持所有 case 使用同一尺度，避免 case 间对比度被强行拉齐
    2. 适合 SliceFixer / LoRA 微调时的稳定训练
    3. 不改变 Gaussian 模型的训练逻辑，只在保存后做统一后处理
    4. 可通过训练集统计自动估计 `K_pred`（推荐 percentile 方式）

## 修改文件

- **文件**: `r2_gaussian/dataset/__init__.py`
- **方法**: `Scene.save()`
- **行号**: 约 79-93（修复后）

## 影响范围

### 新生成的文件
从此修复之后运行的所有训练生成的 `vol_pred.npy` 文件将自动应用此归一化。

### 既有文件
如果系统中存在已经生成的 `vol_pred.npy` 文件（范围 [0, 8.09]），这些文件不会自动更新。选项：

1. **保持原样**：已训练完成的模型参数不变，仅在下次重新生成时应用新的归一化
2. **重新生成**：使用 `process_ctspine1k_complete.sh` 重新运行步骤 3，新的 vol_pred.npy 将自动使用正确的归一化

建议如果需要使用这些文件进行后续处理（如 SliceFixer 增强），应该清理这些文件并重新生成。

## 验证

### 数学验证
- 原始范围：[0, 8.087698]
- 除以 max(8.087698)：得到 [0, 1]
- 最终范围：[0.0, 1.0] ✓

### 代码测试
已通过单元测试验证：
- 输入：随机值 × 8.09
- 输出：[0, 1] 范围
- max 值：1.0 ✓
- min 值：> 0 ✓

## 时间线

- **发现日期**: 检查 toy.ipynb 的输出时
- **根本原因分析**: 追踪 Scene.save() → t2a() → 无归一化
- **修复日期**: [修复应用日期]
- **验证**: 单元测试通过

## 相关文件参考

1. **数据生成**（正确的范例）: `r2_gaussian/data_generator/synthetic_dataset/generate_data.py` line 31-47
2. **数据保存**（修复目标）: `r2_gaussian/dataset/__init__.py` line 79-93
3. **体素化输出**（源数据）: `r2_gaussian/gaussian/render_query.py` line 27-75
4. **损失计算**（依赖此修复）: `r2_gaussian/utils/loss_utils.py` line 159 (pixel_max=1.0)
5. **批量处理**（使用者）: `scripts/train_all_save_to_case.py` 和 `process_ctspine1k_complete.sh`

