import cv2
import numpy as np
from pathlib import Path
import random
from collections import defaultdict
from math import ceil

def copy_paste_augmentation_yolo_v3(
    data_dir='dataset/organized',
    output_dir='dataset/augmented_copypaste_v3',
    target_classes=[1, 2, 3],
    num_augmented=1000,
    paste_per_image=(3, 8),
    min_instance_size=0.02,
    max_scale_ratio=0.5,
    target_ratio=0.3,
    max_multiplier=10,
    padding_ratio=0.15,  # 新增:邊緣padding比例
    class_area_multipliers={1: 1.0, 2: 3.0, 3: 1.5}  # 新增:面積自適應係數
):
    """
    Copy-Paste數據增強 v3 (改進版)
    
    主要改進:
    1. 裁切時包含邊緣區域 (padding)
    2. 針對小物件 (person) 增加額外係數
    3. 優化貼上位置選擇
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    (output_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    
    img_dir = data_path / 'images' / 'train'
    label_dir = data_path / 'labels' / 'train'
    
    print("="*60)
    print("開始 Copy-Paste 數據增強 v3 (改進版)")
    print("="*60)
    print(f"目標類別: {target_classes}")
    print(f"生成數量: {num_augmented}")
    print(f"每張圖貼上: {paste_per_image[0]}-{paste_per_image[1]} 個物件")
    print(f"邊緣 Padding: {padding_ratio*100}%")
    print(f"面積自適應係數: {class_area_multipliers}\n")
    
    # 階段1: 統計各類別數量
    print("階段 1/3: 統計類別分布...")
    class_counts = defaultdict(int)
    all_images = list(img_dir.glob('*.png'))
    
    for img_file in all_images:
        label_file = label_dir / img_file.with_suffix('.txt').name
        if not label_file.exists():
            continue
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    cls = int(parts[0])
                    class_counts[cls] += 1
                except ValueError:
                    continue
    
    N_car = class_counts.get(0, 1)
    
    print(f"\n類別統計:")
    for cls in sorted(class_counts.keys()):
        print(f"  Class {cls}: {class_counts[cls]} 個實例")
    print(f"  多數類 (car) 數量: {N_car}\n")
    
    # 計算每個少數類的 multiplier (加入面積自適應)
    class_multipliers = {}
    for cls in target_classes:
        N_minority = class_counts.get(cls, 1)
        desired_instances = ceil(target_ratio * N_car)
        base_multiplier = max(1, ceil(desired_instances / N_minority))
        
        # 套用面積自適應係數
        area_multiplier = class_area_multipliers.get(cls, 1.0)
        final_multiplier = int(base_multiplier * area_multiplier)
        final_multiplier = min(final_multiplier, max_multiplier * int(area_multiplier))
        
        class_multipliers[cls] = final_multiplier
        print(f"Class {cls}: N={N_minority}, 基礎multiplier={base_multiplier}, "
              f"面積係數={area_multiplier}, 最終={final_multiplier}")
    
    print()
    
    # 階段2: 收集稀有類別實例 (帶padding)
    print("階段 2/3: 收集稀有類別實例 (含邊緣區域)...")
    rare_instances_raw = defaultdict(list)
    
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
            
            if bw_norm < min_instance_size or bh_norm < min_instance_size:
                continue
            
            # 轉換為像素坐標
            xc = xc_norm * w
            yc = yc_norm * h
            bw = bw_norm * w
            bh = bh_norm * h
            
            # ===== 新增: 加入 padding =====
            pad_w = bw * padding_ratio
            pad_h = bh * padding_ratio
            
            x1 = max(0, int(xc - bw / 2 - pad_w))
            y1 = max(0, int(yc - bh / 2 - pad_h))
            x2 = min(w, int(xc + bw / 2 + pad_w))
            y2 = min(h, int(yc + bh / 2 + pad_h))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # 提取實例圖片 (含邊緣)
            instance_img = img[y1:y2, x1:x2].copy()
            
            # 創建漸變mask (中心區域權重高,邊緣漸變到0)
            mask = np.ones((y2-y1, x2-x1), dtype=np.uint8) * 255
            
            # 可選:使用高斯模糊創建更自然的mask
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            
            rare_instances_raw[cls].append({
                'image': instance_img,
                'mask': mask,
                'class': cls,
                'original_size': (x2-x1, y2-y1),
                'area': (x2-x1) * (y2-y1)  # 記錄面積
            })
            collected_count += 1
    
    # 根據 multiplier 複製實例
    rare_instances = defaultdict(list)
    for cls in target_classes:
        if cls not in rare_instances_raw:
            continue
        
        multiplier = class_multipliers[cls]
        original_instances = rare_instances_raw[cls]
        
        for _ in range(multiplier):
            rare_instances[cls].extend(original_instances)
    
    print(f"\n收集並複製完成:")
    total_instances = 0
    for cls in sorted(rare_instances.keys()):
        count = len(rare_instances[cls])
        original_count = len(rare_instances_raw.get(cls, []))
        multiplier = class_multipliers.get(cls, 1)
        total_instances += count
        print(f"  Class {cls}: {original_count} 原始 × {multiplier} = {count} 個實例")
    print(f"  總計: {total_instances} 個實例\n")
    
    if total_instances == 0:
        print("警告: 沒有收集到任何稀有類別實例!")
        return 0
    
    # 階段3: 生成Copy-Paste圖片
    print(f"階段 3/3: 生成 {num_augmented} 張圖片...")
    
    success_count = 0
    for idx in range(num_augmented):
        if idx % 100 == 0:
            print(f"  生成進度: {idx}/{num_augmented}, 成功: {success_count}")
        
        bg_file = random.choice(all_images)
        bg_img = cv2.imread(str(bg_file))
        if bg_img is None:
            continue
        
        bg_img = bg_img.copy()
        h, w = bg_img.shape[:2]
        
        # 讀取背景標註
        label_file = label_dir / bg_file.with_suffix('.txt').name
        new_annotations = []
        existing_boxes = []  # 記錄已有物體位置
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        new_annotations.append(line)
                        parts = line.split()
                        if len(parts) >= 5:
                            # 記錄現有box位置
                            xc, yc, bw, bh = map(float, parts[1:5])
                            existing_boxes.append((xc, yc, bw, bh))
        
        num_paste = random.randint(*paste_per_image)
        pasted_count = 0
        
        for paste_idx in range(num_paste):
            cls = random.choice(target_classes)
            if cls not in rare_instances or len(rare_instances[cls]) == 0:
                continue
            
            instance_data = random.choice(rare_instances[cls])
            inst_img = instance_data['image'].copy()
            inst_mask = instance_data['mask'].copy()
            inst_h, inst_w = inst_img.shape[:2]
            
            # 隨機變換
            scale = random.uniform(0.8, 1.2)
            new_w = max(20, int(inst_w * scale))
            new_h = max(20, int(inst_h * scale))
            
            if new_w >= w * max_scale_ratio or new_h >= h * max_scale_ratio:
                continue
            
            inst_img = cv2.resize(inst_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            inst_mask = cv2.resize(inst_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # 隨機旋轉
            angle = random.uniform(-15, 15)
            center = (new_w / 2, new_h / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            inst_img = cv2.warpAffine(inst_img, M, (new_w, new_h), 
                                     borderMode=cv2.BORDER_CONSTANT, 
                                     borderValue=(0, 0, 0))
            inst_mask = cv2.warpAffine(inst_mask, M, (new_w, new_h), 
                                      borderMode=cv2.BORDER_CONSTANT, 
                                      borderValue=0)
            
            # 隨機翻轉
            if random.random() < 0.5:
                inst_img = cv2.flip(inst_img, 1)
                inst_mask = cv2.flip(inst_mask, 1)
            
            # ===== 新增: 智能位置選擇 (person優先貼在有物體附近) =====
            max_attempts = 30 if cls == 2 else 10  # person多嘗試幾次
            placed = False
            
            for attempt in range(max_attempts):
                max_x = max(1, w - new_w)
                max_y = max(1, h - new_h)
                
                if cls == 2 and len(existing_boxes) > 0 and attempt < max_attempts // 2:
                    # person: 前半次嘗試貼在現有物體附近
                    ref_box = random.choice(existing_boxes)
                    ref_xc, ref_yc = ref_box[0] * w, ref_box[1] * h
                    
                    # 在參考box周圍隨機偏移
                    offset_range = min(w, h) * 0.2
                    x = int(ref_xc + random.uniform(-offset_range, offset_range) - new_w/2)
                    y = int(ref_yc + random.uniform(-offset_range, offset_range) - new_h/2)
                    
                    x = max(0, min(x, max_x))
                    y = max(0, min(y, max_y))
                else:
                    # 完全隨機位置
                    x = random.randint(0, max_x)
                    y = random.randint(0, max_y)
                
                # 檢查是否過度重疊 (簡單版本)
                xc_new = (x + new_w / 2) / w
                yc_new = (y + new_h / 2) / h
                bw_new = new_w / w
                bh_new = new_h / h
                
                # 簡單重疊檢查
                overlap = False
                for ex_xc, ex_yc, ex_bw, ex_bh in existing_boxes:
                    if (abs(xc_new - ex_xc) < (bw_new + ex_bw) / 2 and
                        abs(yc_new - ex_yc) < (bh_new + ex_bh) / 2):
                        overlap = True
                        break
                
                if not overlap or attempt == max_attempts - 1:
                    placed = True
                    break
            
            if not placed:
                continue
            
            # 貼上物件
            try:
                if y + new_h > h or x + new_w > w:
                    continue
                
                roi = bg_img[y:y+new_h, x:x+new_w]
                if roi.shape[:2] != (new_h, new_w):
                    continue
                
                # Alpha混合
                mask_3ch = cv2.cvtColor(inst_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
                
                # ===== 新增: 使用高斯模糊讓邊緣更自然 =====
                mask_3ch = cv2.GaussianBlur(mask_3ch, (5, 5), 1.0)
                
                blended = (roi.astype(np.float32) * (1 - mask_3ch) + 
                          inst_img.astype(np.float32) * mask_3ch).astype(np.uint8)
                
                bg_img[y:y+new_h, x:x+new_w] = blended
                
                # 計算新標註
                xc_pixel = x + new_w / 2
                yc_pixel = y + new_h / 2
                
                xc_norm = max(0, min(1, xc_pixel / w))
                yc_norm = max(0, min(1, yc_pixel / h))
                bw_norm = max(0, min(1, new_w / w))
                bh_norm = max(0, min(1, new_h / h))
                
                new_annotations.append(
                    f"{cls} {xc_norm:.6f} {yc_norm:.6f} {bw_norm:.6f} {bh_norm:.6f}"
                )
                
                # 更新已有box列表
                existing_boxes.append((xc_norm, yc_norm, bw_norm, bh_norm))
                
                pasted_count += 1
                
            except Exception as e:
                continue
        
        if pasted_count > 0 and new_annotations:
            output_filename = f"copypaste_{idx:05d}.png"
            output_img_path = output_path / 'images' / 'train' / output_filename
            output_label_path = output_path / 'labels' / 'train' / f"{output_filename[:-4]}.txt"
            
            cv2.imwrite(str(output_img_path), bg_img)
            
            with open(output_label_path, 'w') as f:
                f.write('\n'.join(new_annotations))
            
            success_count += 1
    
    print(f"\n{'='*60}")
    print("✓ Copy-Paste v3 完成!")
    print(f"{'='*60}")
    print(f"成功生成: {success_count} 張圖片")
    print(f"輸出目錄: {output_dir}")
    
    return success_count

if __name__ == "__main__":
    success = copy_paste_augmentation_yolo_v3(
        data_dir='dataset/organized',
        output_dir='dataset/augmented_copypaste_v3',
        target_classes=[1, 2, 3],
        num_augmented=1200,
        paste_per_image=(4, 10),
        min_instance_size=0.015,
        max_scale_ratio=0.5,
        target_ratio=0.35,
        max_multiplier=10,
        padding_ratio=0.20,  # 15% 邊緣padding
        class_area_multipliers={
            1: 1.0,   # hov - 已經表現不錯
            2: 4.0,   # person - 極小物件需大幅增強
            3: 2.0    # motorcycle - 中等物件適度增強
        }
    )
    
    print(f"\n最終統計: 成功生成 {success} 張增強圖片")