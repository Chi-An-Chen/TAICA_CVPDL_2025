from PIL import Image
import os
from tqdm import tqdm  # ✅ 加入進度條

# 設定路徑
img_dir = "taica-cvpdl-2025-hw-1/train/img"
gt_file = "taica-cvpdl-2025-hw-1/train/gt.txt"
save_label_dir = "taica-cvpdl-2025-hw-1/train/labels"
os.makedirs(save_label_dir, exist_ok=True)

with open(gt_file) as f:
    lines = f.readlines()

# ✅ 在迴圈外加 tqdm
for line in tqdm(lines, desc="Processing images", unit="line"):
    frame, x, y, w, h = line.strip().split(",")
    frame = frame.zfill(8) + ".jpg"
    img_path = os.path.join(img_dir, frame)
    if not os.path.exists(img_path):
        continue

    with Image.open(img_path) as im:
        img_w, img_h = im.size

    # YOLO 需要歸一化座標
    x_center = (float(x) + float(w)/2) / img_w
    y_center = (float(y) + float(h)/2) / img_h
    w_norm = float(w) / img_w
    h_norm = float(h) / img_h

    txt_path = os.path.join(save_label_dir, frame.replace(".jpg", ".txt"))
    with open(txt_path, "a") as ftxt:
        ftxt.write(f"0 {x_center} {y_center} {w_norm} {h_norm}\n")
