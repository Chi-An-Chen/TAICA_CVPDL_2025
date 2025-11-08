import os
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.join(os.path.dirname(__file__)))

from ultralytics import YOLO

# model = YOLO(model="yolov8s-p2-repvgg-sf.yaml")  # yolov8s+repvgg+sf
model = YOLO(model="yolo12m-drone.yaml")  # yolov8s+repvgg+sf

project = "./results"
name    = time.strftime('%Y-%m-%d', time.localtime())

if __name__ == '__main__':
    print(f"Starting training, results will be saved to {os.path.join(project, name)}")
    results = model.train(
        # === 資料與路徑設定 ===
        data='data_aug.yaml',
        project=project,
        name=name,

        resume=True,
        
        epochs=300,
        imgsz=960,
        batch=8,
        
        # 優化器設定
        optimizer='auto',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        dropout=0.05,
        cos_lr=True,
        
        # 數據增強
        seed=144,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,
        degrees=10.0,
        translate=0.1,
        scale=0.25,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        
        # 長尾處理
        cls=4.0,  # 分類損失權重
        box=5.0,
        dfl=1.0,
        
        # 其他設定
        val=True,
        device=0,
        workers=8,
        patience=50,
        save=True,
        save_period=10,
        cache=True,
        plots=True,
        verbose=True,
        
        # 進階設定
        overlap_mask=True,
        deterministic=False,
    )

"""
                   from  n    params  module                                       arguments
  0                  -1  1      1392  ultralytics.nn.modules.conv.Conv             [3, 48, 3, 2]
  1                  -1  1     46464  ultralytics.nn.modules.block.RepVGGBlock     [48, 96, 3, 2]
  2                  -1  1     65088  ultralytics.nn.modules.block.C2f             [96, 96, 1, True]
  3                  -1  1    185088  ultralytics.nn.modules.block.RepVGGBlock     [96, 192, 3, 2]
  4                  -1  2    443904  ultralytics.nn.modules.block.C2f             [192, 192, 2, True]
  5                  -1  1    738816  ultralytics.nn.modules.block.RepVGGBlock     [192, 384, 3, 2]
  6                  -1  2   1772544  ultralytics.nn.modules.block.C2f             [384, 384, 2, True]
  7                  -1  1   2952192  ultralytics.nn.modules.block.RepVGGBlock     [384, 768, 3, 2]
  8                  -1  1   4133376  ultralytics.nn.modules.block.C2f             [768, 768, 1, True]
  9                  -1  1   1476864  ultralytics.nn.modules.block.SPPF            [768, 768, 5]
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 12                  -1  2   2067456  ultralytics.nn.modules.block.C2f             [1152, 384, 2, True]
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 15                  -1  2    517632  ultralytics.nn.modules.block.C2f             [576, 192, 2, True]
 16                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 17             [-1, 2]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 18                  -1  2    129792  ultralytics.nn.modules.block.C2f             [288, 96, 2, True]
 19                  -1  1     83136  ultralytics.nn.modules.conv.Conv             [96, 96, 3, 2]
 20            [-1, 15]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 21                  -1  2    462336  ultralytics.nn.modules.block.C2f             [288, 192, 2, True]
 22                  -1  1    332160  ultralytics.nn.modules.conv.Conv             [192, 192, 3, 2]
 23            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 24                  -1  2   1846272  ultralytics.nn.modules.block.C2f             [576, 384, 2, True]
 25                  -1  1   1327872  ultralytics.nn.modules.conv.Conv             [384, 384, 3, 2]
 26             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 27                  -1  2   7378944  ultralytics.nn.modules.block.C2f             [1152, 768, 2, True]
 28    [18, 21, 24, 27]  1   2573600  ultralytics.nn.modules.head.Detect           [4, [96, 192, 384, 768]]
YOLO12m-drone summary: 200 layers, 28,534,928 parameters, 28,534,912 gradients, 91.8 GFLOPs
"""