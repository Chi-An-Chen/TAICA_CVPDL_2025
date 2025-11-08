import cv2
import numpy as np
from pathlib import Path
import random

def extract_rare_class_instances_yolo(
    data_dir='dataset/organized',
    output_dir='dataset/augmented_crops',
    target_classes=[1, 2, 3],
    crops_per_instance=4,
    min_box_size=0.02,  # 最小框大小（相對於圖片尺寸）
    min_crop_size=80    # 最小crop尺寸（像素）
):
    """
    標準YOLO格式版本
    標註格式: class x_center y_center width height (歸一化值 0-1)
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    (output_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    
    img_dir = data_path / 'images' / 'train'
    label_dir = data_path / 'labels' / 'train'
    
    all_images = sorted(list(img_dir.glob('*.png')))
    crop_count = 0
    class_crop_counts = {cls: 0 for cls in target_classes}
    skipped = {
        'no_rare': 0, 
        'too_small': 0, 
        'read_error': 0, 
        'no_valid_crop': 0, 
        'invalid_anno': 0,
        'no_label': 0
    }
    
    print("開始提取稀有類別實例...")
    print(f"目標類別: {target_classes}")
    print(f"最小物件尺寸: {min_box_size} (歸一化)")
    print(f"最小crop尺寸: {min_crop_size}px")
    print(f"標註格式: class x_center y_center width height (YOLO歸一化格式 0-1)\n")
    
    for idx, img_file in enumerate(all_images):
        if idx % 100 == 0:
            print(f"處理進度: {idx}/{len(all_images)}, 已生成: {crop_count} crops")
        
        # 讀取圖片
        img = cv2.imread(str(img_file))
        if img is None:
            skipped['read_error'] += 1
            continue
        h, w = img.shape[:2]
        
        # 讀取標註
        label_file = label_dir / img_file.with_suffix('.txt').name
        if not label_file.exists():
            skipped['no_label'] += 1
            continue
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            skipped['no_label'] += 1
            continue
        
        # 解析標註（YOLO格式：歸一化值）
        annotations = []
        for line_num, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls = int(parts[0])
                xc_norm = float(parts[1])  # 0-1
                yc_norm = float(parts[2])  # 0-1
                bw_norm = float(parts[3])  # 0-1
                bh_norm = float(parts[4])  # 0-1
                
                # 驗證數值範圍
                if not (0 <= xc_norm <= 1 and 0 <= yc_norm <= 1 and 
                       0 < bw_norm <= 1 and 0 < bh_norm <= 1):
                    skipped['invalid_anno'] += 1
                    continue
                    
                annotations.append((cls, xc_norm, yc_norm, bw_norm, bh_norm))
            except (ValueError, IndexError) as e:
                skipped['invalid_anno'] += 1
                continue
        
        if not annotations:
            continue
        
        # 檢查是否有稀有類別
        rare_instances = [(i, ann) for i, ann in enumerate(annotations) if ann[0] in target_classes]
        
        if not rare_instances:
            skipped['no_rare'] += 1
            continue
        
        # 處理每個稀有類別物件
        for rare_idx, (cls, xc_norm, yc_norm, bw_norm, bh_norm) in rare_instances:
            # 檢查物件尺寸（相對大小）
            if bw_norm < min_box_size or bh_norm < min_box_size:
                skipped['too_small'] += 1
                continue
            
            # 轉換為像素坐標
            xc = xc_norm * w
            yc = yc_norm * h
            bw = bw_norm * w
            bh = bh_norm * h
            
            # 生成多個crops
            successful_crops = 0
            for crop_idx in range(crops_per_instance):
                # 隨機擴展比例
                scale = random.uniform(1.5, 3.0)
                crop_w = max(min_crop_size, int(bw * scale))
                crop_h = max(min_crop_size, int(bh * scale))
                
                # 隨機偏移（讓目標物件不總是在中心）
                offset_range_x = int(bw * 0.3)
                offset_range_y = int(bh * 0.3)
                offset_x = random.randint(-offset_range_x, offset_range_x)
                offset_y = random.randint(-offset_range_y, offset_range_y)
                
                # 計算crop區域（像素坐標）
                crop_xc = xc + offset_x
                crop_yc = yc + offset_y
                
                x1 = int(max(0, crop_xc - crop_w / 2))
                y1 = int(max(0, crop_yc - crop_h / 2))
                x2 = int(min(w, crop_xc + crop_w / 2))
                y2 = int(min(h, crop_yc + crop_h / 2))
                
                # 確保crop大小足夠
                actual_w = x2 - x1
                actual_h = y2 - y1
                
                if actual_w < min_crop_size or actual_h < min_crop_size:
                    skipped['no_valid_crop'] += 1
                    continue
                
                # 裁剪圖片
                cropped_img = img[y1:y2, x1:x2].copy()
                
                # 隨機數據增強
                apply_flip = random.random() < 0.5
                if apply_flip:
                    cropped_img = cv2.flip(cropped_img, 1)
                
                # 轉換所有標註到新坐標系統
                new_annotations = []
                
                for obj_cls, obj_xc_norm, obj_yc_norm, obj_bw_norm, obj_bh_norm in annotations:
                    # 轉換為像素坐標
                    obj_xc = obj_xc_norm * w
                    obj_yc = obj_yc_norm * h
                    obj_bw = obj_bw_norm * w
                    obj_bh = obj_bh_norm * h
                    
                    # 計算物件邊界框
                    obj_x1 = obj_xc - obj_bw / 2
                    obj_y1 = obj_yc - obj_bh / 2
                    obj_x2 = obj_xc + obj_bw / 2
                    obj_y2 = obj_yc + obj_bh / 2
                    
                    # 計算與crop區域的交集
                    inter_x1 = max(x1, obj_x1)
                    inter_y1 = max(y1, obj_y1)
                    inter_x2 = min(x2, obj_x2)
                    inter_y2 = min(y2, obj_y2)
                    
                    # 檢查是否有交集
                    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                        continue
                    
                    # 計算物件可見度（覆蓋率）
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    obj_area = obj_bw * obj_bh
                    visibility = inter_area / obj_area if obj_area > 0 else 0
                    
                    # 只保留可見度大於30%的物件
                    if visibility < 0.3:
                        continue
                    
                    # 計算新的中心點和尺寸（相對於crop，像素坐標）
                    new_xc_pixel = obj_xc - x1
                    new_yc_pixel = obj_yc - y1
                    new_bw_pixel = obj_bw
                    new_bh_pixel = obj_bh
                    
                    # 如果有水平翻轉，調整x坐標
                    if apply_flip:
                        new_xc_pixel = actual_w - new_xc_pixel
                    
                    # 裁剪到crop範圍內
                    new_xc_pixel = max(new_bw_pixel/2, min(actual_w - new_bw_pixel/2, new_xc_pixel))
                    new_yc_pixel = max(new_bh_pixel/2, min(actual_h - new_bh_pixel/2, new_yc_pixel))
                    
                    # 確保尺寸合理
                    new_bw_pixel = min(new_bw_pixel, actual_w)
                    new_bh_pixel = min(new_bh_pixel, actual_h)
                    
                    if new_bw_pixel < 1 or new_bh_pixel < 1:
                        continue
                    
                    # 轉換回YOLO格式（歸一化到0-1）
                    new_xc_norm = new_xc_pixel / actual_w
                    new_yc_norm = new_yc_pixel / actual_h
                    new_bw_norm = new_bw_pixel / actual_w
                    new_bh_norm = new_bh_pixel / actual_h
                    
                    # 確保值在0-1範圍內
                    new_xc_norm = max(0, min(1, new_xc_norm))
                    new_yc_norm = max(0, min(1, new_yc_norm))
                    new_bw_norm = max(0, min(1, new_bw_norm))
                    new_bh_norm = max(0, min(1, new_bh_norm))
                    
                    new_annotations.append(
                        f"{int(obj_cls)} {new_xc_norm:.6f} {new_yc_norm:.6f} {new_bw_norm:.6f} {new_bh_norm:.6f}"
                    )
                
                # 確保至少包含目標稀有類別
                has_target = any(int(ann.split()[0]) == cls for ann in new_annotations)
                if not has_target or len(new_annotations) == 0:
                    continue
                
                # 生成唯一文件名
                crop_filename = f"crop_{img_file.stem}_c{cls}_i{rare_idx}_n{crop_count:05d}.png"
                output_img_path = output_path / 'images' / 'train' / crop_filename
                output_label_path = output_path / 'labels' / 'train' / f"{crop_filename[:-4]}.txt"
                
                # 保存圖片和標註
                cv2.imwrite(str(output_img_path), cropped_img)
                with open(output_label_path, 'w') as f:
                    f.write('\n'.join(new_annotations))
                
                crop_count += 1
                class_crop_counts[cls] += 1
                successful_crops += 1
    
    # 輸出統計結果
    print(f"\n{'='*60}")
    print("✓ 提取完成!")
    print(f"{'='*60}")
    print(f"總共生成: {crop_count} 張crops")
    print(f"\n各類別統計:")
    for cls in sorted(class_crop_counts.keys()):
        print(f"  Class {cls}: {class_crop_counts[cls]} crops")
    
    print(f"\n跳過統計:")
    print(f"  沒有稀有類別: {skipped['no_rare']}")
    print(f"  物件太小: {skipped['too_small']}")
    print(f"  無標註檔案: {skipped['no_label']}")
    print(f"  讀取錯誤: {skipped['read_error']}")
    print(f"  無效crop: {skipped['no_valid_crop']}")
    print(f"  無效標註: {skipped['invalid_anno']}")
    
    return crop_count

if __name__ == "__main__":
    # 空拍視角推薦配置
    crop_count = extract_rare_class_instances_yolo(
        data_dir='dataset/organized',
        output_dir='dataset/augmented_crops',
        target_classes=[1, 2],          # 先處理 hov
        crops_per_instance=2,        # 每個物件生成 4 個 crop
        min_box_size=0.015,          # 空拍物件較小：1.5%
        min_crop_size=80             # 最小 crop 尺寸
    )
    print(f"\n處理完成，共生成 {crop_count} 個crops")