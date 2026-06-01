import numpy as np
import SimpleITK as sitk
from pathlib import Path

# 1. 设置输入与输出路径
base_dir = Path(
    "/root/epfs/test/slicefixer_postprocess_stage3_ssim05_l1zero_iter12000/"
)
npy_files = [
    base_dir / "input_stage3_ssim05_l1zero_iter12000.npy",
    base_dir / "input_stage3_ssim05_l1zero_iter12000_clipped.npy",
    base_dir / "vol_pred_slicefixer_postprocess.npy",
    
]

# 2. 逐个转换
for npy_path in npy_files:
    nii_path = npy_path.with_suffix(".nii.gz")

    vol_data = np.load(npy_path)
    if vol_data.ndim == 4:
        vol_data = np.squeeze(vol_data)

    image = sitk.GetImageFromArray(vol_data)

    # 如有需要可设置体素间距
    # image.SetSpacing([1.0, 1.0, 1.0])

    sitk.WriteImage(image, str(nii_path))
    print(f"转换成功！文件已保存至：{nii_path}")