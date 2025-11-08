import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def multi_domain_augmentation(base_dir, output_base_dir, split='train', 
                               day_range=(1, 603), night_range=(604, 1270)):
    """
    多場景豬隻檢測增強策略
    考慮因素:
    1. 白天/夜間光照差異
    2. 不同攝影機視角
    3. 場景佈局變化
    
    增強策略:
    - 白天(1-603): 彩色保留 + 光照 + 幾何變換
    - 夜間(604-1270): 灰階系列 + 噪聲 + 幾何變換
    """
    input_img_dir = os.path.join(base_dir, split, 'img')
    input_label_dir = os.path.join(base_dir, split, 'labels')
    
    output_img_dir = os.path.join(output_base_dir, split, 'img')
    output_label_dir = os.path.join(output_base_dir, split, 'labels')
    
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    img_files = sorted([f for f in os.listdir(input_img_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"Processing {split} set: {len(img_files)} images")
    print(f"Day: {day_range[0]}-{day_range[1]}, Night: {night_range[0]}-{night_range[1]}")
    
    day_count = night_count = 0
    
    for img_file in tqdm(img_files, desc=f"Augmenting {split}"):
        img_path = os.path.join(input_img_dir, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        base_name = os.path.splitext(img_file)[0]
        ext = os.path.splitext(img_file)[1]
        
        # 提取編號
        import re
        match = re.search(r'(\d+)', base_name)
        img_num = int(match.group(1)) if match else -1
        
        # 檢查 label
        label_file = base_name + '.txt'
        label_path = os.path.join(input_label_dir, label_file)
        label_exists = os.path.exists(label_path)
        
        variants = []
        
        # ========== 白天場景 (彩色) ==========
        if day_range[0] <= img_num <= day_range[1]:
            day_count += 1
            
            # 原圖
            variants.append(('orig', img, None))
            
            # 光照變化
            # variants.append(('bright', adjust_brightness(img, 1.25), None))
            # variants.append(('dark', adjust_brightness(img, 0.75), None))
            
            # 對比度/飽和度
            variants.append(('hicontrast', adjust_contrast(img, 1.3), None))
            variants.append(('desat', reduce_saturation(img, 0.5), None))
            
            # 輕微模糊（模擬運動）
            variants.append(('blur', cv2.GaussianBlur(img, (5,5), 0), None))
            
        # ========== 夜間場景 (灰階) ==========
        elif night_range[0] <= img_num <= night_range[1]:
            night_count += 1
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 原圖
            variants.append(('orig', img, None))
            
            # # 灰階系列
            # variants.append(('gray', cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), None))
            
            # CLAHE 增強
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            variants.append(('clahe', cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR), None))
            
            # 多種 CLAHE 參數
            clahe_strong = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            enhanced_strong = clahe_strong.apply(gray)
            variants.append(('clahe_s', cv2.cvtColor(enhanced_strong, cv2.COLOR_GRAY2BGR), None))
            
            # 低對比度
            low_contrast = cv2.convertScaleAbs(gray, alpha=0.7, beta=20)
            variants.append(('low', cv2.cvtColor(low_contrast, cv2.COLOR_GRAY2BGR), None))
            
            # # 加噪聲
            # noisy = add_noise(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), sigma=12)
            # variants.append(('noisy', noisy, None))
        
        else:
            variants.append(('orig', img, None))
        
        # 保存所有變體
        for suffix, variant_img, transform_type in variants:
            output_img_path = os.path.join(output_img_dir, f"{base_name}_{suffix}{ext}")
            cv2.imwrite(output_img_path, variant_img)
            
            # 複製 label
            if label_exists:
                output_label_path = os.path.join(output_label_dir, f"{base_name}_{suffix}.txt")
                shutil.copy(label_path, output_label_path)


def adjust_brightness(img, factor):
    """調整亮度"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def adjust_contrast(img, factor):
    """調整對比度"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:,:,0] = np.clip(lab[:,:,0] * factor, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def reduce_saturation(img, factor):
    """降低飽和度"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] = hsv[:,:,1] * factor
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# def add_noise(img, sigma=15):
#     """添加高斯噪聲"""
#     noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
#     noisy_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
#     return noisy_img


def preprocess_test_with_clahe(base_dir, output_base_dir, split='test'):
    """
    Test 集預處理 - 根據實際測試集特性選擇
    如果 test 主要是夜間場景，使用 CLAHE
    """
    input_img_dir = os.path.join(base_dir, split, 'img')
    input_label_dir = os.path.join(base_dir, split, 'labels')
    
    output_img_dir = os.path.join(output_base_dir, split, 'img')
    output_label_dir = os.path.join(output_base_dir, split, 'labels')
    
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    img_files = [f for f in os.listdir(input_img_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Processing {split} set: {len(img_files)} images")
    
    for img_file in tqdm(img_files, desc=f"Processing {split}"):
        img_path = os.path.join(input_img_dir, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        base_name = os.path.splitext(img_file)[0]
        ext = os.path.splitext(img_file)[1]
        
        # 根據 test 集特性選擇處理方式
        # 如果 test 是夜間場景，使用 CLAHE
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        output_img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        # 保存
        output_img_path = os.path.join(output_img_dir, img_file)
        cv2.imwrite(output_img_path, output_img)
        
        # 複製 label
        label_file = base_name + '.txt'
        label_path = os.path.join(input_label_dir, label_file)
        if os.path.exists(label_path):
            output_label_path = os.path.join(output_label_dir, label_file)
            shutil.copy(label_path, output_label_path)
    
    print(f"✓ {split} completed: {len(img_files)} images")


if __name__ == '__main__':
    BASE_DIR = 'taica-cvpdl-2025-hw-1'
    OUTPUT_DIR = 'pig_augmented_v2'
    
    # 白天/夜間分界
    DAY_RANGE = (1, 603)
    NIGHT_RANGE = (604, 1270)
    
    # 訓練集增強
    multi_domain_augmentation(
        BASE_DIR, 
        OUTPUT_DIR, 
        split='train',
        day_range=DAY_RANGE,
        night_range=NIGHT_RANGE
    )
    
    
    print("\n" + "="*50)
    print("✅ All preprocessing completed!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*50)