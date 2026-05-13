import numpy as np
import SimpleITK as sitk
from pathlib import Path

# 1. 设置输入与输出路径
base_dir = Path(
    "/home/public/CTSpine1K/data/data-MHD_ctpro_woMask1/volume-covid19-A-0237_ct/"
)
npy_files = [
    base_dir / "vol_pred.npy",
    base_dir / "volume_gt.npy",
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