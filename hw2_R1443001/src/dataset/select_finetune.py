import os
import shutil
from pathlib import Path

# --- 原始資料夾 ---
BASE_DIR = "dataset/data_sahi"  # 原始切片資料
IMAGES_DIR = os.path.join(BASE_DIR, "images", "train")
LABELS_DIR = os.path.join(BASE_DIR, "labels", "train")

# --- 新 finetune 資料夾 ---
FINETUNE_DIR = "dataset/data_sahi_finetune"
FINETUNE_IMAGES_DIR = os.path.join(FINETUNE_DIR, "images", "train")
FINETUNE_LABELS_DIR = os.path.join(FINETUNE_DIR, "labels", "train")

# 建立目錄
os.makedirs(FINETUNE_IMAGES_DIR, exist_ok=True)
os.makedirs(FINETUNE_LABELS_DIR, exist_ok=True)

# 記錄篩選結果
selected_images = []
skipped_images = []

# 遍歷所有標註檔
for label_file in os.listdir(LABELS_DIR):
    if not label_file.endswith(".txt"):
        continue
    label_path = os.path.join(LABELS_DIR, label_file)
    
    with open(label_path, "r") as f:
        lines = f.readlines()
    
    # 判斷是否有非 car 類別 (class_id != "0")
    keep_image = any([line.strip().split()[0] != "0" for line in lines])
    
    if keep_image:
        # 對應圖片檔名
        image_name = Path(label_file).stem
        # 嘗試常見圖片格式
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            image_path = os.path.join(IMAGES_DIR, image_name + ext)
            if os.path.exists(image_path):
                # 複製圖片
                shutil.copy(image_path, os.path.join(FINETUNE_IMAGES_DIR, image_name + ext))
                # 複製標註
                shutil.copy(label_path, os.path.join(FINETUNE_LABELS_DIR, label_file))
                selected_images.append(image_name + ext)
                break
    else:
        skipped_images.append(label_file)

print(f"✅ 篩選完成！")
print(f"  選中圖片數量: {len(selected_images)}")
print(f"  捨去只含 car 的標註: {len(skipped_images)}")
print(f"\n新 finetune 資料集路徑: {FINETUNE_DIR}")
print(f"  ├─images/train/ ({len(selected_images)} 張圖片)")
print(f"  └─labels/train/ ({len(selected_images)} 個標註)")