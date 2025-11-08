import os
import time
import torch
from ultralytics import YOLO

# Load a model
model = YOLO("./yolo12m_drone.yaml")  # build a new model from YAML

project = "./results" # Define a project --> Destination directory for all results
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
        batch=4,

        # warmup
        warmup_epochs=10.0,
        warmup_momentum=0.85,
        
        # 優化器設定
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        dropout=0.0,
        cos_lr=True,
        
        # 數據增強
        seed=144,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.3,  # 重要!
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        shear=2.0,
        perspective=0.0001,
        # rect=True,
        
        # 長尾處理
        cls=0.5,  # 分類損失權重
        box=8.0,
        dfl=1.5,
        
        # 其他設定
        val=True,
        # device=0,
        workers=8,
        patience=50,
        save=True,
        save_period=10,
        cache=False,
        plots=True,
        verbose=True,
        
        # 進階設定
        overlap_mask=True,
        deterministic=False,
        close_mosaic=10,
    )

"""
                   from  n    params  module                                       arguments
  0                  -1  1      1392  ultralytics.nn.modules.conv.Conv             [3, 48, 3, 2]
  1                  -1  1     41664  ultralytics.nn.modules.conv.Conv             [48, 96, 3, 2]
  2                  -1  1     63168  ultralytics.nn.modules.block.C3k2            [96, 192, 1, True, 0.25]
  3                  -1  1    332160  ultralytics.nn.modules.conv.Conv             [192, 192, 3, 2]
  4                  -1  1    250752  ultralytics.nn.modules.block.C3k2            [192, 384, 1, True, 0.25]
  5                  -1  1   1327872  ultralytics.nn.modules.conv.Conv             [384, 384, 3, 2]
  6                  -1  1    837120  ultralytics.nn.modules.block.A2C2f           [384, 384, 1, True, 4]        
  7                  -1  1   1327872  ultralytics.nn.modules.conv.Conv             [384, 384, 3, 2]
  8                  -1  1    837120  ultralytics.nn.modules.block.A2C2f           [384, 384, 1, True, 1]        
  9                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 10             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 11                  -1  1    703104  ultralytics.nn.modules.block.A2C2f           [768, 384, 1, False, -1]
 12                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 13             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 14                  -1  1    213312  ultralytics.nn.modules.block.A2C2f           [768, 192, 1, False, -1]
 15                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 16             [-1, 2]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 17                  -1  1     92736  ultralytics.nn.modules.block.C2f             [384, 96, 1, True]
 18                  -1  1     83136  ultralytics.nn.modules.conv.Conv             [96, 96, 3, 2]
 19            [-1, 14]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 20                  -1  1    167232  ultralytics.nn.modules.block.A2C2f           [288, 192, 1, False, -1]
 21                  -1  1    332160  ultralytics.nn.modules.conv.Conv             [192, 192, 3, 2]
 22            [-1, 11]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 23                  -1  1    666240  ultralytics.nn.modules.block.A2C2f           [576, 384, 1, False, -1]      
 24                  -1  1   1327872  ultralytics.nn.modules.conv.Conv             [384, 384, 3, 2]
 25             [-1, 8]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 26                  -1  1    924672  ultralytics.nn.modules.block.C3k2            [768, 384, 1, True]
 27    [17, 20, 23, 26]  1    930560  ultralytics.nn.modules.head.Detect           [4, [96, 192, 384, 384]]
YOLO12m_drone summary: 281 layers, 10,460,144 parameters, 10,460,128 gradients, 49.7 GFLOPs
"""