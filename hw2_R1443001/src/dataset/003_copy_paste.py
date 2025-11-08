import cv2
import numpy as np
from pathlib import Path
import random
from collections import defaultdict

def copy_paste_augmentation_yolo(
    data_dir='dataset/organized',
    output_dir='dataset/augmented_copypaste',
    target_classes=[1, 2, 3],
    num_augmented=800,
    paste_per_image=(2, 6),
    min_instance_size=0.02,  # 最小實例大小（相對於圖片）
    max_scale_ratio=0.8
):
    """
    Copy-Paste數據增強 (標準YOLO格式)
    標註格式: class x_center y_center width height (歸一化值 0-1)
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    (output_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    
    img_dir = data_path / 'images' / 'train'
    label_dir = data_path / 'labels' / 'train'
    
    print("="*60)
    print("開始 Copy-Paste 數據增強")
    print("="*60)
    print(f"目標類別: {target_classes}")
    print(f"生成數量: {num_augmented}")
    print(f"每張圖貼上: {paste_per_image[0]}-{paste_per_image[1]} 個物件")
    print(f"標註格式: class x_center y_center width height (YOLO歸一化格式 0-1)\n")
    
    # 階段1: 收集稀有類別實例
    print("階段 1/2: 收集稀有類別實例...")
    rare_instances = defaultdict(list)
    all_images = list(img_dir.glob('*.png'))
    
    collected_count = 0
    for img_idx, img_file in enumerate(all_images):
        if img_idx % 100 == 0:
            print(f"  掃描進度: {img_idx}/{len(all_images)}, 已收集: {collected_count}")
        
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        h, w = img.shape[:2]
        
        label_file = label_dir / img_file.with_suffix('.txt').name
        if not label_file.exists():
            continue
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            try:
                cls = int(parts[0])
                xc_norm = float(parts[1])
                yc_norm = float(parts[2])
                bw_norm = float(parts[3])
                bh_norm = float(parts[4])
            except (ValueError, IndexError):
                continue
            
            if cls not in target_classes:
                continue
            
            # 檢查尺寸（相對大小）
            if bw_norm < min_instance_size or bh_norm < min_instance_size:
                continue
            
            # 轉換為像素坐標
            xc = xc_norm * w
            yc = yc_norm * h
            bw = bw_norm * w
            bh = bh_norm * h
            
            # 計算邊界框（像素坐標）
            x1 = max(0, int(xc - bw / 2))
            y1 = max(0, int(yc - bh / 2))
            x2 = min(w, int(xc + bw / 2))
            y2 = min(h, int(yc + bh / 2))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # 提取實例圖片
            instance_img = img[y1:y2, x1:x2].copy()
            
            # 創建簡單的矩形mask
            mask = np.ones((y2-y1, x2-x1), dtype=np.uint8) * 255
            
            rare_instances[cls].append({
                'image': instance_img,
                'mask': mask,
                'class': cls,
                'original_size': (x2-x1, y2-y1)
            })
            collected_count += 1
    
    print(f"\n收集完成:")
    total_instances = 0
    for cls in sorted(rare_instances.keys()):
        count = len(rare_instances[cls])
        total_instances += count
        print(f"  Class {cls}: {count} 個實例")
    print(f"  總計: {total_instances} 個實例\n")
    
    if total_instances == 0:
        print("警告: 沒有收集到任何稀有類別實例！")
        return 0
    
    # 階段2: 生成Copy-Paste圖片
    print(f"階段 2/2: 生成 {num_augmented} 張 Copy-Paste 圖片...")
    
    success_count = 0
    for idx in range(num_augmented):
        if idx % 100 == 0:
            print(f"  生成進度: {idx}/{num_augmented}, 成功: {success_count}")
        
        # 隨機選擇背景圖片
        bg_file = random.choice(all_images)
        bg_img = cv2.imread(str(bg_file))
        if bg_img is None:
            continue
        
        bg_img = bg_img.copy()
        h, w = bg_img.shape[:2]
        
        # 讀取背景圖片的原始標註
        label_file = label_dir / bg_file.with_suffix('.txt').name
        new_annotations = []
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        new_annotations.append(line)
        
        # 隨機決定貼上多少個物件
        num_paste = random.randint(*paste_per_image)
        pasted_count = 0
        
        for paste_idx in range(num_paste):
            # 隨機選擇類別
            cls = random.choice(target_classes)
            if cls not in rare_instances or len(rare_instances[cls]) == 0:
                continue
            
            # 隨機選擇該類別的一個實例
            instance_data = random.choice(rare_instances[cls])
            inst_img = instance_data['image'].copy()
            inst_mask = instance_data['mask'].copy()
            inst_h, inst_w = inst_img.shape[:2]
            
            # 隨機縮放
            scale = random.uniform(0.6, 1.8)
            new_w = max(20, int(inst_w * scale))
            new_h = max(20, int(inst_h * scale))
            
            # 確保不會太大（不超過背景圖的一定比例）
            if new_w >= w * max_scale_ratio or new_h >= h * max_scale_ratio:
                continue
            
            # 調整大小
            inst_img = cv2.resize(inst_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            inst_mask = cv2.resize(inst_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # 隨機旋轉（小角度）
            if random.random() < 0.3:
                angle = random.uniform(-15, 15)
                center = (new_w / 2, new_h / 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                inst_img = cv2.warpAffine(inst_img, M, (new_w, new_h), 
                                         borderMode=cv2.BORDER_CONSTANT, 
                                         borderValue=(0, 0, 0))
                inst_mask = cv2.warpAffine(inst_mask, M, (new_w, new_h), 
                                          borderMode=cv2.BORDER_CONSTANT, 
                                          borderValue=0)
            
            # 隨機水平翻轉
            if random.random() < 0.5:
                inst_img = cv2.flip(inst_img, 1)
                inst_mask = cv2.flip(inst_mask, 1)
            
            # 隨機選擇貼上位置
            max_x = max(1, w - new_w)
            max_y = max(1, h - new_h)
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            
            # 使用mask進行融合
            try:
                # 確保座標在範圍內
                if y + new_h > h or x + new_w > w:
                    continue
                
                # 獲取ROI
                roi = bg_img[y:y+new_h, x:x+new_w]
                
                if roi.shape[:2] != (new_h, new_w):
                    continue
                
                # 創建3通道mask
                mask_3ch = cv2.cvtColor(inst_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
                
                # Alpha混合
                blended = (roi.astype(np.float32) * (1 - mask_3ch) + 
                          inst_img.astype(np.float32) * mask_3ch).astype(np.uint8)
                
                bg_img[y:y+new_h, x:x+new_w] = blended
                
                # 計算新標註（YOLO歸一化格式）
                xc_pixel = x + new_w / 2
                yc_pixel = y + new_h / 2
                
                # 轉換為歸一化坐標 (0-1)
                xc_norm = xc_pixel / w
                yc_norm = yc_pixel / h
                bw_norm = new_w / w
                bh_norm = new_h / h
                
                # 確保值在0-1範圍內
                xc_norm = max(0, min(1, xc_norm))
                yc_norm = max(0, min(1, yc_norm))
                bw_norm = max(0, min(1, bw_norm))
                bh_norm = max(0, min(1, bh_norm))
                
                new_annotations.append(
                    f"{cls} {xc_norm:.6f} {yc_norm:.6f} {bw_norm:.6f} {bh_norm:.6f}"
                )
                
                pasted_count += 1
                
            except Exception as e:
                continue
        
        # 只有成功貼上至少一個物件才保存
        if pasted_count > 0 and new_annotations:
            output_filename = f"copypaste_{idx:05d}.png"
            output_img_path = output_path / 'images' / 'train' / output_filename
            output_label_path = output_path / 'labels' / 'train' / f"{output_filename[:-4]}.txt"
            
            # 保存圖片
            cv2.imwrite(str(output_img_path), bg_img)
            
            # 保存標註
            with open(output_label_path, 'w') as f:
                f.write('\n'.join(new_annotations))
            
            success_count += 1
    
    print(f"\n{'='*60}")
    print("✓ Copy-Paste 完成!")
    print(f"{'='*60}")
    print(f"成功生成: {success_count} 張圖片")
    print(f"輸出目錄: {output_dir}")
    
    return success_count

if __name__ == "__main__":
    # 空拍視角推薦配置
    success = copy_paste_augmentation_yolo(
        data_dir='dataset/organized',
        output_dir='dataset/augmented_copypaste',
        target_classes=[1, 2, 3],          # 針對 hov
        num_augmented=1000,           # 空拍：保守生成 300 張
        paste_per_image=(3, 6),      # 每張 3-5 個物件
        min_instance_size=0.02,      # 空拍物件較小：2%
        max_scale_ratio=0.5          # 空拍：最大 50%
    )
    
    print(f"\n最終統計: 成功生成 {success} 張增強圖片")