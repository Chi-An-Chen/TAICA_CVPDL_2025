import os
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 1. 設定路徑 ---
BASE_DIR = "dataset/data_sahi"  # 處理後的資料集路徑
SPLIT = "val"  # 要檢查的資料集: "train" 或 "val"

# --- 2. 視覺化參數 ---
NUM_SAMPLES = 9  # 要顯示的樣本數量 (建議 4, 9, 16 等完全平方數)
RANDOM_SEED = 42  # 隨機種子

# 顏色列表 (為不同類別設定不同顏色)
COLORS = [
    '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', 
    '#00FFFF', '#FFA500', '#800080', '#FFC0CB', '#A52A2A'
]

# --- 3. 讀取 YOLO 標註 ---
def read_yolo_annotations(label_path):
    """
    讀取 YOLO 格式標註
    返回: [(class_id, center_x, center_y, width, height), ...]
    """
    annotations = []
    
    if not os.path.exists(label_path):
        return annotations
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                center_x = float(parts[1])
                center_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                annotations.append((class_id, center_x, center_y, width, height))
    
    return annotations

# --- 4. 在圖片上繪製邊界框 ---
def draw_boxes_on_image(image, annotations):
    """
    在圖片上繪製 YOLO 標註框
    """
    img_width, img_height = image.size
    draw = ImageDraw.Draw(image)
    
    # 嘗試載入字體
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except:
        font = ImageFont.load_default()
    
    for class_id, center_x, center_y, width, height in annotations:
        # 轉換 YOLO 格式到像素座標
        x_center = center_x * img_width
        y_center = center_y * img_height
        box_width = width * img_width
        box_height = height * img_height
        
        # 計算邊界框座標
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)
        
        # 選擇顏色
        color = COLORS[class_id % len(COLORS)]
        
        # 繪製邊界框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # 繪製類別標籤
        label = f"Class {class_id}"
        
        # 計算文字背景大小
        bbox = draw.textbbox((x1, y1), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 繪製文字背景
        draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], 
                      fill=color)
        
        # 繪製文字
        draw.text((x1 + 2, y1 - text_height - 2), label, fill='white', font=font)
    
    return image

# --- 5. 使用 Matplotlib 顯示多張圖片 ---
def visualize_samples_matplotlib(samples, grid_size):
    """
    使用 matplotlib 顯示多張樣本圖片
    """
    rows = int(grid_size ** 0.5)
    cols = rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    fig.suptitle(f'切片資料視覺化檢查 - {SPLIT.upper()} Set', fontsize=20, y=0.995)
    
    # 如果只有一個樣本
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)
    
    for idx, (image, annotations, filename) in enumerate(samples):
        if idx >= rows * cols:
            break
        
        row = idx // cols
        col = idx % cols
        ax = axes[row][col]
        
        # 繪製邊界框
        img_with_boxes = draw_boxes_on_image(image.copy(), annotations)
        
        # 顯示圖片
        ax.imshow(img_with_boxes)
        ax.set_title(f'{filename}\nobjects: {len(annotations)}', fontsize=10)
        ax.axis('off')
    
    # 隱藏多餘的子圖
    for idx in range(len(samples), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row][col].axis('off')
    
    plt.tight_layout()
    plt.show()

# --- 6. 單張圖片詳細檢查 ---
def visualize_single_sample(image, annotations, filename):
    """
    顯示單張圖片的詳細資訊
    """
    img_width, img_height = image.size
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # 繪製圖片
    ax.imshow(image)
    
    # 繪製每個邊界框
    for idx, (class_id, center_x, center_y, width, height) in enumerate(annotations):
        # 轉換到像素座標
        x_center = center_x * img_width
        y_center = center_y * img_height
        box_width = width * img_width
        box_height = height * img_height
        
        x1 = x_center - box_width / 2
        y1 = y_center - box_height / 2
        
        # 選擇顏色
        color = COLORS[class_id % len(COLORS)]
        
        # 繪製矩形
        rect = patches.Rectangle((x1, y1), box_width, box_height, 
                                 linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # 添加標籤
        ax.text(x1, y1 - 5, f'Class {class_id}', 
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
               fontsize=10, color='white', weight='bold')
    
    ax.set_title(f'{filename}\nimg size: {img_width}x{img_height}, objects: {len(annotations)}', 
                fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

# --- 7. 統計資訊 ---
def print_dataset_statistics(image_dir, label_dir):
    """
    打印資料集統計資訊
    """
    print("\n" + "=" * 60)
    print(f"資料集統計資訊 - {SPLIT.upper()}")
    print("=" * 60)
    
    # 統計圖片和標註檔案
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    label_files = [f for f in os.listdir(label_dir) 
                   if f.lower().endswith('.txt')]
    
    print(f"圖片數量: {len(image_files)}")
    print(f"標註數量: {len(label_files)}")
    
    # 統計物件數量
    total_objects = 0
    class_counts = {}
    empty_labels = 0
    
    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)
        annotations = read_yolo_annotations(label_path)
        
        if len(annotations) == 0:
            empty_labels += 1
            continue
        
        total_objects += len(annotations)
        
        for class_id, _, _, _, _ in annotations:
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    print(f"\n物件統計:")
    print(f"  總物件數: {total_objects}")
    print(f"  空標註檔案: {empty_labels}")
    print(f"  平均每張: {total_objects / len(label_files):.2f} 個物件")
    
    print(f"\n各類別統計:")
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = (count / total_objects) * 100
        print(f"  Class {class_id}: {count} 個 ({percentage:.1f}%)")
    
    print("=" * 60)

# --- 8. 主程式 ---
if __name__ == "__main__":
    # 設定路徑
    image_dir = os.path.join(BASE_DIR, "images", SPLIT)
    label_dir = os.path.join(BASE_DIR, "labels", SPLIT)
    
    # 檢查目錄是否存在
    if not os.path.exists(image_dir):
        print(f"錯誤: 找不到圖片目錄 {image_dir}")
        exit(1)
    if not os.path.exists(label_dir):
        print(f"錯誤: 找不到標註目錄 {label_dir}")
        exit(1)
    
    # 打印統計資訊
    print_dataset_statistics(image_dir, label_dir)
    
    # 取得所有圖片檔案
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if len(image_files) == 0:
        print("錯誤: 沒有找到任何圖片!")
        exit(1)
    
    # 隨機選擇樣本
    random.seed(RANDOM_SEED)
    sample_files = random.sample(image_files, min(NUM_SAMPLES, len(image_files)))
    
    # 載入樣本
    samples = []
    for image_file in sample_files:
        image_path = os.path.join(image_dir, image_file)
        image_name = Path(image_file).stem
        label_file = image_name + ".txt"
        label_path = os.path.join(label_dir, label_file)
        
        # 載入圖片
        image = Image.open(image_path)
        
        # 載入標註
        annotations = read_yolo_annotations(label_path)
        
        samples.append((image, annotations, image_file))
    
    print(f"\n隨機選擇了 {len(samples)} 個樣本進行視覺化...")
    
    # 顯示多張圖片
    visualize_samples_matplotlib(samples, NUM_SAMPLES)
    
    # 詢問是否要查看單張圖片的詳細資訊
    print("\n" + "=" * 60)
    print("如果要查看單張圖片詳細資訊,請執行以下程式碼:")
    print("visualize_single_sample(samples[0][0], samples[0][1], samples[0][2])")
    print("=" * 60)