"""
Author: Chi-An Chen
Date: 2025-11-27
Version: 2.1
Description: 使用 batch + CUDA 計算圖片 mean / std (含 resize, 適用於不同大小圖片)
"""

import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

def load_image_tensor(path, resize_size=512):
    """讀取圖片並轉成 [C,H,W] 的 float tensor，並統一 resize"""
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype("float32") / 255.0
    t = torch.from_numpy(arr).permute(2,0,1)  # [H,W,C] → [C,H,W]

    # Resize：所有圖片調整到 [C, resize_size, resize_size]
    t = F.interpolate(t.unsqueeze(0), size=(resize_size, resize_size), mode='bilinear', align_corners=False)
    return t.squeeze(0)  # [C,H,W]


def calculate_mean_std_cuda(data_path, batch_size=32, resize_size=512, device="cuda"):
    """使用 batch + CUDA 計算圖片 mean/std (所有圖片自動 resize)"""
    img_paths = list(Path(data_path).rglob("*.jpg")) + \
                list(Path(data_path).rglob("*.jpeg")) + \
                list(Path(data_path).rglob("*.JPG")) + \
                list(Path(data_path).rglob("*.JPEG"))

    if not img_paths:
        raise ValueError(f"在 {data_path} 找不到任何圖片")

    print(f"找到 {len(img_paths)} 張圖片")

    pixel_sum = torch.zeros(3, device=device)
    pixel_sq_sum = torch.zeros(3, device=device)
    total_pixels = 0

    batch = []

    for p in tqdm(img_paths, desc="處理圖片", unit="img"):
        try:
            batch.append(load_image_tensor(p, resize_size=resize_size))
        except:
            continue

        if len(batch) == batch_size:
            batch_tensor = torch.stack(batch).to(device)  # [B,C,H,W]
            B, C, H, W = batch_tensor.shape

            pixel_sum += batch_tensor.sum(dim=[0, 2, 3])
            pixel_sq_sum += (batch_tensor ** 2).sum(dim=[0, 2, 3])
            total_pixels += B * H * W

            batch = []

    if len(batch) > 0:
        batch_tensor = torch.stack(batch).to(device)
        B, C, H, W = batch_tensor.shape
        pixel_sum += batch_tensor.sum(dim=[0, 2, 3])
        pixel_sq_sum += (batch_tensor ** 2).sum(dim=[0, 2, 3])
        total_pixels += B * H * W

    mean = pixel_sum / total_pixels
    std = torch.sqrt(pixel_sq_sum / total_pixels - mean ** 2)

    print("\nMean (RGB):", mean.cpu().numpy())
    print("Std  (RGB):", std.cpu().numpy())

    print("\n用於 transforms.Normalize:")
    print(f"mean={[round(v,3) for v in mean.cpu().numpy()]}")
    print(f"std={[round(v,3) for v in std.cpu().numpy()]}")
    
    return mean.cpu().numpy(), std.cpu().numpy()


if __name__ == "__main__":

    root_dir = "./TAICA期末資料集"
    data_path = f"{root_dir}/citrus"

    calculate_mean_std_cuda(
        data_path,
        batch_size=256,   # 根據 GPU 大小調整
        resize_size=512, # 建議 256, 512 均可
        device="cuda"
    )
