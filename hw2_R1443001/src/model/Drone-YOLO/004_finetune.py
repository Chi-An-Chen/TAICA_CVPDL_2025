import os
import time
import torch
import sys
import os

from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load a model
model = YOLO("yolov8s-p2-repvgg-sf.yaml").load("results_finetune/2025-10-23/weights/epoch60.pt")  # build from YAML and transfer weights

project = "./results_finetune" # Define a project --> Destination directory for all results
name    = time.strftime('%Y-%m-%d', time.localtime()) # Define subdirectory for this specific training

# Train the model
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

if __name__ == '__main__':
    print(f"Starting training, results will be saved to {os.path.join(project, name)}")
    results = model.train(
        # === 資料與路徑設定 ===
        data='data.yaml',
        project=project,
        name=name,

        freeze=9,
        resume=True,
        
        epochs=100,
        imgsz=960,
        batch=16,
        
        # 優化器設定
        optimizer='auto',
        lr0=0.0001,
        lrf=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        dropout=0.0,
        cos_lr=True,
        
        # 數據增強
        seed=144,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
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
        patience=30,
        save=True,
        save_period=10,
        cache=False,
        plots=True,
        
        # 進階設定
        overlap_mask=True,
        deterministic=False,
    )

"""
                   from  n    params  module                                       arguments
  0                  -1  1      1856  ultralytics.nn.modules.conv.Conv             [3, 64, 3, 2]
  1                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]
  2                  -1  1    111872  ultralytics.nn.modules.block.C3k2            [128, 256, 1, True, 0.25]
  3                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]
  4                  -1  1    444928  ultralytics.nn.modules.block.C3k2            [256, 512, 1, True, 0.25]
  5                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]
  6                  -1  2   2689536  ultralytics.nn.modules.block.A2C2f           [512, 512, 2, True, 4]        
  7                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]
  8                  -1  2   2689536  ultralytics.nn.modules.block.A2C2f           [512, 512, 2, True, 1]        
  9                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 10             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 11                  -1  1   1248768  ultralytics.nn.modules.block.A2C2f           [1024, 512, 1, False, -1]
 12                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 13             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 14                  -1  1    378624  ultralytics.nn.modules.block.A2C2f           [1024, 256, 1, False, -1]
 15                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]
 16            [-1, 11]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 17                  -1  1   1183232  ultralytics.nn.modules.block.A2C2f           [768, 512, 1, False, -1]
 18                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]
 19             [-1, 8]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 20                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]
 21        [14, 17, 20]  1   1414108  ultralytics.nn.modules.head.Detect           [4, [256, 512, 512]]
YOLO12m summary: 292 layers, 20,140,572 parameters, 20,140,556 gradients, 67.8 GFLOPs
"""