import os
import time
import torch
from ultralytics import YOLO

# Load a model
model = YOLO("yolo12m.yaml")  # build a new model from YAML

project = "./results_square" # Define a project --> Destination directory for all results
name    = time.strftime('%Y-%m-%d', time.localtime()) # Define subdirectory for this specific training

# Train the model
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

if __name__ == '__main__':
    print(f"Starting training, results will be saved to {os.path.join(project, name)}")
    results = model.train(
        # === 資料與路徑設定 ===
        data='data_aug.yaml',
        project=project,
        name=name,
        
        epochs=500,
        imgsz=960,
        batch=8,

        # warmup
        warmup_epochs=10.0,
        warmup_momentum=0.85,
        
        # 優化器設定
        optimizer='auto',
        lr0=0.01,
        lrf=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        dropout=0.05,
        cos_lr=True,
        
        # 數據增強
        seed=144,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,  # 重要!
        degrees=10.0,
        translate=0.1,
        scale=0.3,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        # rect=True,
        
        # 長尾處理
        cls=3.5,  # 分類損失權重
        box=5.5,
        dfl=1.0,
        
        # 其他設定
        val=True,
        device=0,
        workers=8,
        patience=100,
        save=True,
        save_period=10,
        cache=False,
        plots=True,
        
        # 進階設定
        overlap_mask=True,
        deterministic=False,
    )