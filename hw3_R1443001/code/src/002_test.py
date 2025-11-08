"""
Author: Chi-An Chen
Date: 2025-11-08
Description: Inference
"""
import os
import torch

from tqdm import tqdm
from torchvision.utils import save_image, make_grid

from model.model import DDPM, UNetModel

def load_model(checkpoint_path: str, device: str = "cuda:0"):
    """載入訓練好的模型"""
    # 載入 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 創建模型 (確保和 training 設置相同)
    unet = UNetModel(
        in_channels=3,
        model_channels=64,
        out_channels=3,
        num_res_blocks=2,
        channel_mult=(1, 2, 4),
        dropout=0.1,
        use_scale_shift_norm=True,
    )
    
    ddpm = DDPM(
        eps_model=unet,
        betas=(1e-4, 0.02),
        n_T=1000
    )
    
    # 載入模型權重
    ddpm.load_state_dict(checkpoint['model_state_dict'])
    ddpm.to(device)
    ddpm.eval()
    
    # 獲取 mean 和 std
    mean = checkpoint.get('mean', torch.tensor([0.1309, 0.1309, 0.1309]))
    std = checkpoint.get('std', torch.tensor([0.2893, 0.2893, 0.2893]))
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"Mean: {mean.tolist()}")
    print(f"Std: {std.tolist()}")
    
    # 如果有 EMA 權重,載入它
    if 'ema_shadow' in checkpoint:
        print("Loading EMA weights...")
        ema_shadow = checkpoint['ema_shadow']
        for name, param in ddpm.named_parameters():
            if name in ema_shadow:
                param.data = ema_shadow[name]
    
    return ddpm, mean, std

def denormalize_and_resize(images: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, target_size=(28, 28)):
    """
    反正規化並調整圖片大小到 28x28
    
    Args:
        images: 正規化的圖片 tensor [B, C, H, W]
        mean: 正規化使用的 mean
        std: 正規化使用的 std
        target_size: 目標大小
    
    Returns:
        處理後的圖片 tensor [B, C, 28, 28]
    """
    # 反正規化
    mean = mean.view(1, 3, 1, 1).to(images.device)
    std = std.view(1, 3, 1, 1).to(images.device)
    images = images * std + mean
    
    # Resize 到 28x28
    if images.shape[-2:] != target_size:
        images = torch.nn.functional.interpolate(
            images, 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        )
    
    # Clip 到 [0, 1]
    images = torch.clamp(images, 0, 1)
    
    return images

def generate_fid_images(
    ddpm: DDPM,
    mean: torch.Tensor,
    std: torch.Tensor,
    num_images: int = 10000,
    batch_size: int = 100,
    save_dir: str = "./generated_mnist/",
    device: str = "cuda:0"
):
    """
    生成指定數量的圖片用於 FID 計算
    
    Args:
        ddpm: 訓練好的 DDPM 模型
        mean: 資料集 mean
        std: 資料集 std
        num_images: 要生成的圖片數量
        batch_size: 每批生成的圖片數量
        save_dir: 圖片保存目錄
        device: 設備
    """
    os.makedirs(save_dir, exist_ok=True)
    
    ddpm.eval()
    num_batches = (num_images + batch_size - 1) // batch_size
    image_count = 0
    
    print(f"Generating {num_images} images for FID calculation...")
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Generating images"):
            # 計算當前批次應該生成的圖片數量
            current_batch_size = min(batch_size, num_images - image_count)
            
            # 生成圖片 (32x32)
            images = ddpm.sample(current_batch_size, (3, 32, 32), device)
            
            # 反正規化並 resize 到 28x28
            images = denormalize_and_resize(images, mean, std, target_size=(28, 28))
            
            # 保存每張圖片
            for i in range(current_batch_size):
                img = images[i]
                # 轉換為灰度圖 (取 RGB 的平均值)
                img_gray = img.mean(dim=0, keepdim=True)
                
                # 保存為 PNG
                save_path = os.path.join(save_dir, f"{image_count:05d}.png")
                save_image(img_gray, save_path)
                image_count += 1
    
    print(f"Successfully generated {image_count} images in {save_dir}")

def visualize_diffusion_process(
    ddpm: DDPM,
    mean: torch.Tensor,
    std: torch.Tensor,
    num_samples: int = 8,
    num_steps: int = 8,
    save_path: str = "./diffusion_process.png",
    device: str = "cuda:0"
):
    """
    視覺化擴散過程 (已修正採樣公式)
    """
    ddpm.eval()
    
    n_T = ddpm.n_T
    
    # 計算要記錄的時間步: 從純噪聲到完全去噪
    if num_steps == 1:
        timesteps_to_record = [0]
    else:
        # 均勻分布的時間步
        # [1000, 857, 714, 571, 428, 285, 142, 0] (假設 n_T=1000, num_steps=8)
        timesteps_to_record = [int(n_T * (1 - i / (num_steps - 1))) for i in range(num_steps)]
        # 確保 t=0 絕對被包含
        timesteps_to_record[-1] = 0 
        # 確保 t=T 在最前面 (用於初始噪聲)
        timesteps_to_record[0] = n_T

    print(f"Recording timesteps: {timesteps_to_record}")
    
    all_images = []
    
    with torch.no_grad():
        # 1. 初始化噪聲 (x_T)
        x_i = torch.randn(num_samples, 3, 32, 32).to(device)
        
        # 2. 記錄初始噪聲 (t=1000)
        images_to_save = denormalize_and_resize(
            x_i.clone(), mean, std, target_size=(28, 28)
        )
        all_images.append(images_to_save)
        print(f"Recorded t={n_T} (initial noise)")
        
        # 用於追蹤下一個要記錄的時間步
        next_record_idx = 1
        
        # 3. 反向擴散過程 (從 t=T 迴圈到 t=1)
        for i in range(n_T, 0, -1):
            # 使用整數 timestep（與訓練、sample 一致）
            t = torch.tensor([i], device=device).repeat(num_samples)

            # predict eps (model expects integer timesteps)
            eps = ddpm.eps_model(x_i, t)

            # 當 i>1 時需要加入噪聲
            if i > 1:
                z = torch.randn_like(x_i)
            else:
                z = torch.zeros_like(x_i)

            # 與 sample() 完全相同的更新
            x_i = (
                ddpm.oneover_sqrta[i] * (x_i - eps * ddpm.mab_over_sqrtmab[i])
                + ddpm.sqrt_beta_t[i] * z
            )

            # 記錄要保存的 timestep（如果到達 timesteps_to_record）
            if next_record_idx < len(timesteps_to_record):
                target_t = timesteps_to_record[next_record_idx]
                # 這裡比較 i-1 與 target_t（保持你原本的邏輯）
                if (i - 1) <= target_t:
                    images_to_save = denormalize_and_resize(
                        x_i.clone(), mean, std, target_size=(28, 28)
                    )
                    all_images.append(images_to_save)
                    print(f"Recorded t={i-1}")
                    next_record_idx += 1
                    while next_record_idx < len(timesteps_to_record) and timesteps_to_record[next_record_idx] >= (i-1):
                        next_record_idx += 1

    # 4. 組織圖片為網格
    grid_images = []

    # 重新組織順序：先 timestep 再 sample 變成 sample 為列、time 為行
    for step_idx in range(len(all_images)):          # 時間順序
        for sample_idx in range(num_samples):        # 每個 sample
            grid_images.append(all_images[step_idx][sample_idx])

    grid_images = torch.stack(grid_images)

    grid = make_grid(grid_images, nrow=num_samples, padding=2, pad_value=0.0)
    
    # 5. 保存
    save_image(grid, save_path)
    print(f"Diffusion process visualization saved to {save_path}")
    print(f"Total steps recorded: {len(all_images)}")
    
    return timesteps_to_record

def main():
    """主函數"""
    # 設定參數
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoint_path = "./checkpoints/ddpm_mnist_ema.pth"  # 或使用特定的 checkpoint
    num_images = 10000
    print(f"Using device: {device}")
    
    # 載入模型
    ddpm, mean, std = load_model(checkpoint_path, device)
    
    # 1. 生成 10000 張圖片用於 FID 計算
    print("\n" + "="*50)
    print(f"Task 1: Generating {num_images} images for FID calculation")
    print("="*50)
    generate_fid_images(
        ddpm=ddpm,
        mean=mean,
        std=std,
        num_images=num_images,
        batch_size=100,
        save_dir="./generated_mnist/",
        device=device
    )
    
    # 2. 視覺化擴散過程
    print("\n" + "="*50)
    print("Task 2: Visualizing diffusion process")
    print("="*50)
    timesteps = visualize_diffusion_process(
        ddpm=ddpm,
        mean=mean,
        std=std,
        num_samples=8,
        num_steps=8,
        save_path="./diffusion_process.png",
        device=device
    )
    
    print(f"\nRecorded timesteps: {timesteps}")
    print("\nAll tasks completed!")


if __name__ == "__main__":
    main()


    