"""
Author: Chi-An Chen
Date: 2025-11-08
Description: Training DDPM model
"""
import os
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image, make_grid

from model.model import DDPM, UNetModel


class CustomMNISTDataset(Dataset):
    """自定義 MNIST 資料集載入器"""
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0  # 返回 dummy label


def calculate_dataset_statistics(dataset_path: str, batch_size: int = 128):
    """計算資料集的 mean 和 std"""
    print("[INFO] Calculating Mean and Std of Dataset ...")
    
    # 臨時的轉換,只做 resize 和 ToTensor
    temp_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    
    dataset = CustomMNISTDataset(dataset_path, transform=temp_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 計算 mean 和 std
    channels_sum = torch.zeros(3)
    channels_squared_sum = torch.zeros(3)
    num_batches = 0
    
    for images, _ in tqdm(dataloader, desc="Calculating"):
        channels_sum += torch.mean(images, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(images ** 2, dim=[0, 2, 3])
        num_batches += 1
    
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    
    print(f"[INFO] Mean: {mean.tolist()}")
    print(f"[INFO] Std: {std.tolist()}")
    
    return mean, std


class EMAHelper:
    """指數移動平均 (EMA) 輔助類"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # 初始化 shadow 參數
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """更新 EMA 參數"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """將 EMA 參數應用到模型"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """恢復原始參數"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class WarmupCosineScheduler:
    """帶預熱的 Cosine 學習率排程器"""
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Warmup 階段:線性增長
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine 衰減階段
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


def train_mnist(
    n_epoch: int = 100,
    device: str = "cuda:0",
    dataset_path: str = "./dataset/mnist/",
    batch_size: int = 128,
    base_lr: float = 2e-4,
    warmup_epochs: int = 5,
    use_ema: bool = True,
    ema_decay: float = 0.9999,
    calculate_stats: bool = True
) -> None:
    """
    在自定義 MNIST 資料集上訓練 DDPM
    
    Args:
        n_epoch: 訓練輪數
        device: 訓練設備
        dataset_path: 資料集路徑
        batch_size: 批次大小
        base_lr: 基礎學習率
        warmup_epochs: 預熱輪數
        use_ema: 是否使用 EMA
        ema_decay: EMA 衰減率
        calculate_stats: 是否計算資料集統計資訊
    """
    # 創建 UNet 模型 - 修改為 3 通道輸入
    unet = UNetModel(
        in_channels=3,  # RGB 圖片
        model_channels=64,
        out_channels=3,  # RGB 輸出
        num_res_blocks=2,
        channel_mult=(1, 2, 4),  # 32 -> 16 -> 8,特徵提取更友好
        dropout=0.1,
        use_scale_shift_norm=True,
    )
    
    ddpm = DDPM(
        eps_model=unet,
        betas=(1e-4, 0.02),
        n_T=1000,
        # schedule_type="cosine"
    )
    ddpm.to(device)

    try:
        from torchinfo import summary
        x = torch.zeros(1, 3, 32, 32).to(device)
        t = torch.zeros(1, dtype=torch.long).to(device)
        summary(unet, input_data=(x, t))
    except:
        pass

    # 計算或使用預設的 mean 和 std
    if calculate_stats:
        mean, std = calculate_dataset_statistics(dataset_path, batch_size)
    else:
        # 如果你已經計算過,可以直接填入數值
        mean = torch.tensor([0.5, 0.5, 0.5])  # 替換為實際值
        std = torch.tensor([0.5, 0.5, 0.5])   # 替換為實際值
    
    mean_tensor = mean.view(1, 3, 1, 1).to(device)
    std_tensor = std.view(1, 3, 1, 1).to(device)
    
    # 數據預處理 - 調整為 32x32
    tf = transforms.Compose([
        transforms.Resize((32, 32)),  # 從 28x28 調整為 32x32
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(), std.tolist())
    ])
    
    # 載入自定義資料集
    dataset = CustomMNISTDataset(dataset_path, transform=tf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=4, pin_memory=True)
    
    # 優化器
    optim = torch.optim.AdamW(ddpm.parameters(), lr=base_lr, weight_decay=1e-4)
    
    # 學習率排程器
    total_steps = n_epoch * len(dataloader)
    warmup_steps = warmup_epochs * len(dataloader)
    scheduler = WarmupCosineScheduler(
        optim, 
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        base_lr=base_lr,
        min_lr=1e-6
    )
    
    # EMA 輔助器
    ema_helper = EMAHelper(ddpm, decay=ema_decay) if use_ema else None
    
    # 創建保存目錄
    os.makedirs("./contents", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)
    
    # 訓練迴圈
    print("\n========== Start Training ==========\n")

    global_step = 0
    best_loss = float('inf')

    for epoch in range(n_epoch):
        ddpm.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epoch}")
        loss_ema = None
        
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss = ddpm(x)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            
            optim.step()
            
            # 更新學習率
            current_lr = scheduler.step()
            
            # 更新 EMA
            if ema_helper is not None:
                ema_helper.update()
            
            # 計算移動平均損失
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'EMA loss': f'{loss_ema:.4f}',
                'lr': f'{current_lr:.2e}'
            })
            
            global_step += 1
        
        # 每個 epoch 結束後生成樣本並保存模型
        ddpm.eval()
        if (epoch + 1) % 5 == 0 or (epoch + 1) == 1:
            with torch.no_grad():
                # 使用原始權重生成樣本
                xh = ddpm.sample(64, (3, 32, 32), device)
                xh = xh * std_tensor + mean_tensor
                xh = torch.clamp(xh, 0, 1)
                grid = make_grid(xh, nrow=8)
                save_image(grid, f"./contents/ddpm_sample_epoch{epoch+1}.png")
                
                # 如果使用 EMA,也用 EMA 權重生成樣本
                if ema_helper is not None:
                    ema_helper.apply_shadow()
                    xh_ema = ddpm.sample(64, (3, 32, 32), device)
                    xh_ema = xh_ema * std_tensor + mean_tensor
                    xh_ema = torch.clamp(xh_ema, 0, 1)
                    grid_ema = make_grid(xh_ema, nrow=8)
                    save_image(grid_ema, f"./contents/ddpm_sample_epoch{epoch+1}_ema.png")
                    ema_helper.restore()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': ddpm.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': loss_ema,
            'mean': mean,
            'std': std
        }
        
        if ema_helper is not None:
            checkpoint['ema_shadow'] = ema_helper.shadow

        if loss_ema<best_loss:
            best_loss = loss_ema
            torch.save(checkpoint, f"./checkpoints/ddpm_mnist_best.pth")
            print(f'[INFO] Best loss checkpoint saved !')
        else:
            print(f'[INFO] Best loss = {best_loss:.6f}')
        
        if use_ema:
            if (epoch + 1) % 10 == 0:
                torch.save(checkpoint, f"./checkpoints/ddpm_mnist_ema.pth")
                print(f'[INFO] EMA checkpoint saved !')
        


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"[INFO] Training using {device}")

    train_mnist(
        n_epoch=200,
        device=device,
        dataset_path="./dataset/MNIST/",
        batch_size=256,
        base_lr=2e-4,
        warmup_epochs=10,
        use_ema=True,
        ema_decay=0.9999,
        calculate_stats=True
    )
    
