import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm


class DatasetReorganizer:
    """YOLO 數據集重組工具"""
    
    def __init__(self, base_dir, val_ratio=0.1, seed=123):
        """
        Args:
            base_dir: 原始資料根目錄
            val_ratio: 驗證集比例
            seed: 隨機種子
        """
        self.base_dir = Path(base_dir)
        self.val_ratio = val_ratio
        self.seed = seed
        random.seed(seed)
        
    def split_train_val(self, train_img_dir, train_label_dir, val_img_dir, val_label_dir):
        """
        從訓練集中分割出驗證集
        
        Args:
            train_img_dir: 訓練圖片目錄
            train_label_dir: 訓練標註目錄
            val_img_dir: 驗證圖片目標目錄
            val_label_dir: 驗證標註目標目錄
        """
        train_img_dir = Path(train_img_dir)
        train_label_dir = Path(train_label_dir)
        val_img_dir = Path(val_img_dir)
        val_label_dir = Path(val_label_dir)
        
        # 建立驗證集資料夾
        val_img_dir.mkdir(parents=True, exist_ok=True)
        val_label_dir.mkdir(parents=True, exist_ok=True)
        
        # 獲取所有圖片
        all_images = [f for f in os.listdir(train_img_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not all_images:
            print(f"⚠️  Warning: No images found in {train_img_dir}")
            return
        
        # 計算驗證集大小並隨機抽樣
        val_size = int(len(all_images) * self.val_ratio)
        val_images = random.sample(all_images, val_size)
        
        print(f"📊 Splitting dataset:")
        print(f"   Total: {len(all_images)} images")
        print(f"   Val:   {val_size} images ({self.val_ratio*100:.1f}%)")
        print(f"   Train: {len(all_images)-val_size} images ({(1-self.val_ratio)*100:.1f}%)")
        
        # 移動驗證集
        moved_count = 0
        missing_labels = []
        
        for img_name in tqdm(val_images, desc="Moving to validation set"):
            # 生成對應的標註檔名
            label_name = Path(img_name).stem + '.txt'
            
            img_src = train_img_dir / img_name
            img_dst = val_img_dir / img_name
            label_src = train_label_dir / label_name
            label_dst = val_label_dir / label_name
            
            # 移動圖片
            if img_src.exists():
                shutil.move(str(img_src), str(img_dst))
                moved_count += 1
            
            # 移動標註
            if label_src.exists():
                shutil.move(str(label_src), str(label_dst))
            else:
                missing_labels.append(img_name)
        
        print(f"✓ Moved {moved_count} images to validation set")
        
        if missing_labels:
            print(f"⚠️  Warning: {len(missing_labels)} images missing labels")
            if len(missing_labels) <= 5:
                for name in missing_labels:
                    print(f"   - {name}")
    
    def reorganize_to_yolo_structure(self, source_dirs, output_base_dir):
        """
        重組為 YOLO 標準結構
        
        Args:
            source_dirs: dict，格式 {'train': {'img': path, 'labels': path}, 'val': {...}}
            output_base_dir: 輸出根目錄
        """
        output_base = Path(output_base_dir)
        
        print(f"\n📁 Reorganizing to YOLO structure:")
        print(f"   Output: {output_base}")
        
        for split, dirs in source_dirs.items():
            img_src = Path(dirs['img'])
            label_src = Path(dirs['labels'])
            
            img_dst = output_base / 'images' / split
            label_dst = output_base / 'labels' / split
            
            # 建立目標資料夾
            img_dst.mkdir(parents=True, exist_ok=True)
            label_dst.mkdir(parents=True, exist_ok=True)
            
            # 移動圖片
            if img_src.exists():
                img_files = list(img_src.glob('*'))
                for img_file in tqdm(img_files, desc=f"Moving {split} images"):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        shutil.move(str(img_file), str(img_dst / img_file.name))
            
            # 移動標註
            if label_src.exists():
                label_files = list(label_src.glob('*.txt'))
                for label_file in tqdm(label_files, desc=f"Moving {split} labels"):
                    shutil.move(str(label_file), str(label_dst / label_file.name))
        
        print("✓ Reorganization completed")
    
    def cleanup_empty_dirs(self, *dirs):
        """清理空資料夾"""
        for d in dirs:
            d = Path(d)
            if d.exists() and not any(d.iterdir()):
                d.rmdir()
                print(f"🗑️  Removed empty directory: {d}")
    
    def print_summary(self, output_base_dir):
        """打印數據集摘要"""
        output_base = Path(output_base_dir)
        
        print(f"\n{'='*60}")
        print("📊 Dataset Summary")
        print(f"{'='*60}")
        
        for split in ['train', 'val', 'test']:
            img_dir = output_base / 'images' / split
            label_dir = output_base / 'labels' / split
            
            if img_dir.exists():
                img_count = len(list(img_dir.glob('*.[jp][pn]g')))
                label_count = len(list(label_dir.glob('*.txt'))) if label_dir.exists() else 0
                
                print(f"{split.capitalize():8s}: {img_count:5d} images, {label_count:5d} labels")
        
        print(f"{'='*60}\n")


def main():
    """主函數 - 完整流程範例"""
    
    # ==================== 配置區 ====================
    
    # 原始資料路徑
    SOURCE_TRAIN_IMG = "pig_augmented_v2/train/img"
    SOURCE_TRAIN_LABEL = "pig_augmented_v2/train/labels"
    
    # 臨時驗證集路徑（會在過程中創建）
    TEMP_VAL_IMG = "pig_augmented_v2/val/img"
    TEMP_VAL_LABEL = "pig_augmented_v2/val/labels"
    
    # 最終輸出路徑（YOLO 標準結構）
    OUTPUT_BASE_DIR = "dataset_final_v2"
    
    # 驗證集比例
    VAL_RATIO = 0.1
    
    # 隨機種子（用於可重現的分割）
    SEED = 123
    
    # 類別名稱
    CLASS_NAMES = ['pig']
    
    # 測試集路徑（如果有的話）
    TEST_IMG_DIR = None  # 例如: "images/test"
    
    # ================================================
    
    # 初始化重組器
    reorganizer = DatasetReorganizer(
        base_dir=OUTPUT_BASE_DIR,
        val_ratio=VAL_RATIO,
        seed=SEED
    )
    
    # 步驟1: 分割訓練集和驗證集
    print("Step 1: Splitting train/val")
    print("-" * 60)
    reorganizer.split_train_val(
        train_img_dir=SOURCE_TRAIN_IMG,
        train_label_dir=SOURCE_TRAIN_LABEL,
        val_img_dir=TEMP_VAL_IMG,
        val_label_dir=TEMP_VAL_LABEL
    )
    
    # 步驟2: 重組為 YOLO 標準結構
    print("\nStep 2: Reorganizing to YOLO structure")
    print("-" * 60)
    
    source_structure = {
        'train': {
            'img': SOURCE_TRAIN_IMG,
            'labels': SOURCE_TRAIN_LABEL
        },
        'val': {
            'img': TEMP_VAL_IMG,
            'labels': TEMP_VAL_LABEL
        }
    }
    
    reorganizer.reorganize_to_yolo_structure(
        source_dirs=source_structure,
        output_base_dir=OUTPUT_BASE_DIR
    )
    
    # 步驟3: 清理臨時資料夾
    print("\nStep 3: Cleaning up")
    print("-" * 60)
    reorganizer.cleanup_empty_dirs(
        SOURCE_TRAIN_IMG,
        SOURCE_TRAIN_LABEL,
        TEMP_VAL_IMG,
        TEMP_VAL_LABEL,
        "pig_augmented_v2/train",
        "pig_augmented_v2/val"
    )
    
    # 打印摘要
    reorganizer.print_summary(OUTPUT_BASE_DIR)
    
    # 完成提示
    print("🎉 All done!")


if __name__ == '__main__':
    main()