import os
import time
import torch
from ultralytics import YOLO

# Load a model
model = YOLO("results/2025-10-25/weights/best.pt")  # build from YAML and transfer weights 

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

        freeze=8,
        
        epochs=200,
        imgsz=960,
        batch=4,
        
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
        device=0,
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
"""
                   from  n    params  module                                       arguments
  0                  -1  1       928  ultralytics.nn.modules.conv.Conv             [3, 32, 3, 2]
  1                  -1  1     20736  ultralytics.nn.modules.block.RepVGGBlock     [32, 64, 3, 2]
  2                  -1  1     29056  ultralytics.nn.modules.block.C2f             [64, 64, 1, True]
  3                  -1  1     82432  ultralytics.nn.modules.block.RepVGGBlock     [64, 128, 3, 2]
  4                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]
  5                  -1  1    328704  ultralytics.nn.modules.block.RepVGGBlock     [128, 256, 3, 2]
  6                  -1  2    788480  ultralytics.nn.modules.block.C2f             [256, 256, 2, True]
  7                  -1  1   1312768  ultralytics.nn.modules.block.RepVGGBlock     [256, 512, 3, 2]
  8                  -1  1   1838080  ultralytics.nn.modules.block.C2f             [512, 512, 1, True]
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 11                   4  1      1280  ultralytics.nn.modules.conv.DWConv           [128, 64, 3, 2]
 12         [-1, 6, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 13                  -1  1    607744  ultralytics.nn.modules.block.C2f             [832, 256, 1]
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 15                   2  1       640  ultralytics.nn.modules.conv.DWConv           [64, 32, 3, 2]
 16         [-1, 4, 14]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 17                  -1  1    152320  ultralytics.nn.modules.block.C2f             [416, 128, 1]
 18                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 19                   0  1       320  ultralytics.nn.modules.conv.DWConv           [32, 16, 3, 2]
 20         [-1, 2, 18]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 21                  -1  1     38272  ultralytics.nn.modules.block.C2f             [208, 64, 1]
 22                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]
 23            [-1, 17]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 24                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]
 25                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]
 26            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 27                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]
 28                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]
 29             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 30                  -1  1   1969152  ultralytics.nn.modules.block.C2f             [768, 512, 1]
 31    [21, 24, 27, 30]  1   1420576  ultralytics.nn.modules.head.Detect           [4, [64, 128, 256, 512]]
YOLOv8s-p2-repvgg-sf summary: 182 layers, 10,837,760 parameters, 10,837,744 gradients, 37.6 GFLOPs
"""