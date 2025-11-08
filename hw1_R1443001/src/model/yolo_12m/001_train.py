import os
import time
import torch
from ultralytics import YOLO

# Load a model
model = YOLO("yolo12m.yaml")  # build a new model from YAML

project = "./results" # Define a project --> Destination directory for all results
name    = time.strftime('%Y-%m-%d', time.localtime()) # Define subdirectory for this specific training

# Train the model
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

if __name__ == '__main__':
    print(f"Starting training, results will be saved to {os.path.join(project, name)}")
    results = model.train(
        # === 資料與路徑設定 ===
        data="data_aug.yaml",
        project=project,
        name=name,
        
        # === 基本訓練參數 ===
        epochs=35,
        batch=16,
        imgsz=640,
        device=device,
        workers=8,
        
        # === 優化器與學習率 ===
        optimizer="AdamW",
        lr0=0.01,
        cos_lr=True,
        
        # === 正則化參數 ===
        dropout=0.05,
        erasing=0.0,
        
        # === 其他設定 ===
        seed=123,
        patience=20,
        mosaic=0.5,
        mixup=0.5,
        hsv_h=0.015,       # 色調增強（白天場景）
        hsv_s=0.3,         # 飽和度（減弱，因為夜間無色彩）
        hsv_v=0.4,         # 亮度（重要！）
        degrees=10,        # 輕微旋轉（模擬視角變化）
        translate=0.1,     # 平移
        scale=0.3,         # 縮放
        fliplr=0.5,        # 水平翻轉
        
        # === 建議新增的參數 ===
        # 學習率相關
        lrf=0.01,              # 最終學習率 (lr0 * lrf)
        warmup_epochs=3.0,     # 學習率熱身輪數
        
        # 權重衰減
        weight_decay=0.0005,   # L2正則化
        
        # 驗證與保存
        val=True,              # 啟用驗證 (默認True，但明確寫出)
        save_period=5,        # 每10個epoch保存checkpoint
        
        # 輸出控制
        plots=True,            # 生成訓練圖表
    )