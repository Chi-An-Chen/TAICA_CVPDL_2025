# merge_datasets.py
from pathlib import Path
import shutil

def merge_augmented_data():
    """合併增強數據到原始數據集"""
    
    # 定義路徑
    original_dir = Path('dataset/organized')
    # crops_dir = Path('dataset/augmented_crops')
    copypaste_dir = Path('dataset/augmented_copypaste_v3')
    output_dir = Path('dataset/final_train_v2')
    
    # 創建輸出目錄
    (output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    
    total_copied = 0
    
    # 複製原始數據
    print("複製原始數據...")
    for img_file in (original_dir / 'images' / 'train').glob('*.png'):
        shutil.copy(img_file, output_dir / 'images' / 'train' / img_file.name)
        label_file = original_dir / 'labels' / 'train' / f"{img_file.stem}.txt"
        if label_file.exists():
            shutil.copy(label_file, output_dir / 'labels' / 'train' / f"{img_file.stem}.txt")
        total_copied += 1
    print(f"  原始: {total_copied} 張")
    
    # # 複製 Crop 數據
    # crop_count = 0
    # if crops_dir.exists():
    #     print("複製 Crop 增強數據...")
    #     for img_file in (crops_dir / 'images' / 'train').glob('*.png'):
    #         shutil.copy(img_file, output_dir / 'images' / 'train' / img_file.name)
    #         label_file = crops_dir / 'labels' / 'train' / f"{img_file.stem}.txt"
    #         if label_file.exists():
    #             shutil.copy(label_file, output_dir / 'labels' / 'train' / f"{img_file.stem}.txt")
    #         crop_count += 1
    #     print(f"  Crops: {crop_count} 張")
    
    # 複製 Copy-Paste 數據
    copypaste_count = 0
    if copypaste_dir.exists():
        print("複製 Copy-Paste 數據...")
        for img_file in (copypaste_dir / 'images' / 'train').glob('*.png'):
            shutil.copy(img_file, output_dir / 'images' / 'train' / img_file.name)
            label_file = copypaste_dir / 'labels' / 'train' / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy(label_file, output_dir / 'labels' / 'train' / f"{img_file.stem}.txt")
            copypaste_count += 1
        print(f"  Copy-Paste: {copypaste_count} 張")
    
    total = total_copied + copypaste_count
    print(f"\n✓ 合併完成！")
    print(f"總計: {total} 張圖片")
    print(f"輸出目錄: {output_dir}")

if __name__ == "__main__":
    merge_augmented_data()