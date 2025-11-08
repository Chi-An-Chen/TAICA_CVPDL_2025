import shutil
from pathlib import Path
import random
from PIL import Image

def normalize_coordinates(x1, y1, w, h, img_width, img_height):
    """
    將邊界框坐標從像素轉換為YOLO格式（中心點x, y和寬高，範圍0-1）
    
    Args:
        x1, y1: 邊界框左上角坐標（像素）
        w, h: 邊界框寬度和高度（像素）
        img_width, img_height: 圖片寬度和高度
    
    Returns:
        cx, cy, w, h: YOLO格式的歸一化坐標
    """
    # 計算中心點
    cx = x1 + w / 2
    cy = y1 + h / 2
    
    # 歸一化到0-1範圍
    cx_norm = cx / img_width
    cy_norm = cy / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    
    # 確保值在0-1範圍內
    cx_norm = max(0, min(1, cx_norm))
    cy_norm = max(0, min(1, cy_norm))
    w_norm = max(0, min(1, w_norm))
    h_norm = max(0, min(1, h_norm))
    
    return cx_norm, cy_norm, w_norm, h_norm


def organize_dataset(source_dir='CVPDL_hw2/train', target_dir='dataset/organized'):
    """
    將原始數據組織成YOLO格式，並同時轉換標註檔中的逗號分隔格式，
    以及將座標歸一化。
    
    原始格式: <class>,<x1>,<y1>,<width>,<height>
    YOLO格式: <class> <cx> <cy> <w> <h> (所有值歸一化到0-1)
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 創建目錄結構
    (target_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (target_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (target_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (target_path / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    # 獲取所有圖片文件
    all_images = sorted(list(source_path.glob('*.png')))
    print(f"找到 {len(all_images)} 張圖片")
    
    # 分析類別分佈
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # car, motorcycle, person, hov
    invalid_boxes = 0
    
    for img_file in all_images:
        txt_file = img_file.with_suffix('.txt')
        if txt_file.exists():
            with open(txt_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # 支援逗號或空白分隔
                    parts = line.replace(',', ' ').split()
                    try:
                        cls = int(parts[0])
                        w = float(parts[3])
                        h = float(parts[4])
                        if w > 0 and h > 0:
                            class_counts[cls] += 1
                        else:
                            invalid_boxes += 1
                    except (ValueError, IndexError):
                        print(f"⚠️ 無法解析標註行: {line} (於 {txt_file.name})")
                        continue
    
    print("\n類別分佈:")
    print(f"  Class 0 (car):        {class_counts[0]}")
    print(f"  Class 1 (motorcycle): {class_counts[1]}")
    print(f"  Class 2 (person):     {class_counts[2]}")
    print(f"  Class 3 (hov):        {class_counts[3]}")
    if invalid_boxes > 0:
        print(f"  ⚠️ 無效邊界框 (w=0 或 h=0): {invalid_boxes}")
    
    random.seed(144)
    random.shuffle(all_images)
    split_idx = int(len(all_images) * 0.95)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    print(f"\n分割數據集:")
    print(f"  訓練集: {len(train_images)} 張")
    print(f"  驗證集: {len(val_images)} 張")
    
    # 複製文件 + 標註轉換 + 座標歸一化
    total_converted = 0
    total_skipped = 0
    
    for img_list, split in [(train_images, 'train'), (val_images, 'val')]:
        for img_file in img_list:
            # 複製圖片
            shutil.copy(
                img_file,
                target_path / 'images' / split / img_file.name
            )
            
            # 獲取圖片尺寸
            try:
                with Image.open(img_file) as img:
                    img_width, img_height = img.size
            except Exception as e:
                print(f"⚠️ 無法讀取圖片尺寸: {img_file.name} - {e}")
                continue
            
            # 處理標註檔
            txt_file = img_file.with_suffix('.txt')
            if txt_file.exists():
                with open(txt_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # 轉換逗號為空白並進行座標歸一化
                converted_lines = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.replace(',', ' ').split()
                    
                    if len(parts) < 5:
                        print(f"⚠️ 標註格式錯誤: {line} (於 {txt_file.name})")
                        total_skipped += 1
                        continue
                    
                    try:
                        cls = int(parts[0])
                        x1, y1, w, h = map(float, parts[1:5])
                        
                        # 跳過無效的邊界框（寬高為0或負數）
                        if w <= 0 or h <= 0:
                            print(f"⚠️ 跳過無效邊界框 (w={w}, h={h}): {line} (於 {txt_file.name})")
                            total_skipped += 1
                            continue
                        
                        # 歸一化座標
                        cx, cy, w_norm, h_norm = normalize_coordinates(
                            x1, y1, w, h, img_width, img_height
                        )
                        
                        # 格式: class cx cy w h
                        normalized_line = f"{cls} {cx:.6f} {cy:.6f} {w_norm:.6f} {h_norm:.6f}"
                        converted_lines.append(normalized_line + '\n')
                        total_converted += 1
                        
                    except (ValueError, IndexError) as e:
                        print(f"⚠️ 座標轉換失敗: {line} (於 {txt_file.name}) - {e}")
                        total_skipped += 1
                        continue
                
                # 儲存為新的標註檔
                out_label_path = target_path / 'labels' / split / txt_file.name
                with open(out_label_path, 'w', encoding='utf-8') as f:
                    f.writelines(converted_lines)
    
    print(f"\n✓ 數據組織、標註格式轉換與座標歸一化完成!")
    print(f"  成功轉換: {total_converted} 個邊界框")
    print(f'平均每張圖片檢測物件數: {total_converted / len(all_images)}')
    if total_skipped > 0:
        print(f"  跳過無效: {total_skipped} 個邊界框")
    
    return target_path, class_counts


# 執行組織
if __name__ == "__main__":
    organized_path, class_counts = organize_dataset()