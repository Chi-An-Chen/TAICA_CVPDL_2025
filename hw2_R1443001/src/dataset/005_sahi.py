import os
from pathlib import Path
from PIL import Image
import random

# --- 1. 定義路徑 ---
BASE_INPUT_DIR = "dataset/organized"
BASE_OUTPUT_DIR = "dataset/data_sahi"

# --- 2. 切片參數設定 ---
SLICE_HEIGHT = 640
SLICE_WIDTH = 640
OVERLAP_HEIGHT_RATIO = 0.25
OVERLAP_WIDTH_RATIO = 0.25

# --- 3. 切分比例設定 ---
TRAIN_RATIO = 0.9  # 90% 訓練集, 10% 驗證集

# --- 4. 物件可見度閾值設定 ---
MIN_AREA_RATIO = 0.6  # 物件在切片中的最小可見面積比例(相對於原始物件)
MIN_BBOX_SIZE = 10    # 裁切後邊界框的最小邊長(像素)

# --- 5. 計算切片座標 ---
def get_slice_bboxes(image_width, image_height, slice_height, slice_width, 
                     overlap_height_ratio, overlap_width_ratio):
    """
    計算所有切片的座標
    返回: [(y_min, y_max, x_min, x_max), ...]
    """
    slice_bboxes = []
    
    # 計算步長(考慮重疊)
    y_overlap = int(slice_height * overlap_height_ratio)
    x_overlap = int(slice_width * overlap_width_ratio)
    y_step = slice_height - y_overlap
    x_step = slice_width - x_overlap
    
    # 計算 y 方向的切片
    y_min = 0
    while y_min < image_height:
        y_max = min(y_min + slice_height, image_height)
        
        # 計算 x 方向的切片
        x_min = 0
        while x_min < image_width:
            x_max = min(x_min + slice_width, image_width)
            
            slice_bboxes.append((y_min, y_max, x_min, x_max))
            
            # 如果已經到達右邊界,跳出
            if x_max >= image_width:
                break
            x_min += x_step
        
        # 如果已經到達底部邊界,跳出
        if y_max >= image_height:
            break
        y_min += y_step
    
    return slice_bboxes

# --- 6. 檢查物件在切片中的可見度 ---
def calculate_bbox_visibility(bbox_pixel, slice_bbox):
    """
    計算物件在切片中的可見度
    返回: (可見面積比例, 交集邊界框像素座標)
    """
    obj_x_min, obj_y_min, obj_x_max, obj_y_max = bbox_pixel
    slice_y_min, slice_y_max, slice_x_min, slice_x_max = slice_bbox
    
    # 計算物件原始面積
    obj_width = obj_x_max - obj_x_min
    obj_height = obj_y_max - obj_y_min
    obj_area = obj_width * obj_height
    
    # 計算交集邊界
    inter_x_min = max(obj_x_min, slice_x_min)
    inter_y_min = max(obj_y_min, slice_y_min)
    inter_x_max = min(obj_x_max, slice_x_max)
    inter_y_max = min(obj_y_max, slice_y_max)
    
    # 檢查是否有交集
    if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
        return 0.0, None
    
    # 計算交集面積
    inter_width = inter_x_max - inter_x_min
    inter_height = inter_y_max - inter_y_min
    inter_area = inter_width * inter_height
    
    # 計算可見度(相對於原始物件面積的比例)
    visibility = inter_area / obj_area if obj_area > 0 else 0.0
    
    # 返回可見度和交集邊界框
    return visibility, (inter_x_min, inter_y_min, inter_x_max, inter_y_max)

# --- 7. 轉換 YOLO 標註到切片座標(保留部分物件) ---
def convert_yolo_to_slice(yolo_annotations, image_width, image_height, 
                         slice_y_min, slice_y_max, slice_x_min, slice_x_max,
                         min_area_ratio=0.1, min_bbox_size=10):
    """
    將原始 YOLO 標註轉換到切片座標系統
    保留所有與切片有足夠交集的物件(允許裁切)
    
    參數:
        min_area_ratio: 最小可見面積比例(相對於原始物件)
        min_bbox_size: 裁切後邊界框的最小邊長(像素)
    """
    slice_width = slice_x_max - slice_x_min
    slice_height = slice_y_max - slice_y_min
    slice_annotations = []
    kept_objects = 0
    total_objects = 0
    
    for line in yolo_annotations:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        
        total_objects += 1
        
        class_id = parts[0]
        center_x = float(parts[1]) * image_width
        center_y = float(parts[2]) * image_height
        bbox_width = float(parts[3]) * image_width
        bbox_height = float(parts[4]) * image_height
        
        # 計算原始邊界框座標(像素)
        x_min = center_x - bbox_width / 2
        y_min = center_y - bbox_height / 2
        x_max = center_x + bbox_width / 2
        y_max = center_y + bbox_height / 2
        
        # 計算物件在切片中的可見度
        visibility, intersect_bbox = calculate_bbox_visibility(
            (x_min, y_min, x_max, y_max),
            (slice_y_min, slice_y_max, slice_x_min, slice_x_max)
        )
        
        # 如果可見度太低,跳過此物件
        if visibility < min_area_ratio or intersect_bbox is None:
            continue
        
        # 取得交集邊界框
        inter_x_min, inter_y_min, inter_x_max, inter_y_max = intersect_bbox
        
        # 檢查裁切後的邊界框大小是否足夠
        clipped_width = inter_x_max - inter_x_min
        clipped_height = inter_y_max - inter_y_min
        
        if clipped_width < min_bbox_size or clipped_height < min_bbox_size:
            continue
        
        # 轉換到切片座標系統(相對於切片左上角)
        new_x_min = inter_x_min - slice_x_min
        new_y_min = inter_y_min - slice_y_min
        new_x_max = inter_x_max - slice_x_min
        new_y_max = inter_y_max - slice_y_min
        
        # 轉換回 YOLO 格式(歸一化到切片大小)
        new_center_x = (new_x_min + new_x_max) / 2 / slice_width
        new_center_y = (new_y_min + new_y_max) / 2 / slice_height
        new_width = (new_x_max - new_x_min) / slice_width
        new_height = (new_y_max - new_y_min) / slice_height
        
        # 確保座標在 [0, 1] 範圍內
        new_center_x = max(0, min(1, new_center_x))
        new_center_y = max(0, min(1, new_center_y))
        new_width = max(0, min(1, new_width))
        new_height = max(0, min(1, new_height))
        
        # 最後檢查: 確保轉換後的寬高不為0
        if new_width > 0 and new_height > 0:
            slice_annotations.append(
                f"{class_id} {new_center_x:.6f} {new_center_y:.6f} {new_width:.6f} {new_height:.6f}"
            )
            kept_objects += 1
    
    return slice_annotations, kept_objects, total_objects

# --- 8. 處理單張圖片並返回切片資料 ---
def slice_single_image(image_path, label_path, slice_height, slice_width, 
                      overlap_height_ratio, overlap_width_ratio,
                      min_area_ratio, min_bbox_size):
    """
    切片單張圖片及其標註,返回切片資料列表
    允許裁切物件,但確保足夠的可見度
    返回: [(slice_image, slice_annotations), ...], 統計資訊
    """
    # 讀取圖片
    image = Image.open(image_path)
    image_width, image_height = image.size
    image_name = Path(image_path).stem
    image_ext = Path(image_path).suffix
    
    # 讀取 YOLO 標註
    yolo_annotations = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            yolo_annotations = f.readlines()
    
    # 取得所有切片座標
    slice_bboxes = get_slice_bboxes(
        image_width, image_height, 
        slice_height, slice_width,
        overlap_height_ratio, overlap_width_ratio
    )
    
    slices_data = []
    total_kept_objects = 0
    total_original_objects = 0
    
    # 處理每個切片
    for idx, (y_min, y_max, x_min, x_max) in enumerate(slice_bboxes):
        # 裁剪圖片
        slice_image = image.crop((x_min, y_min, x_max, y_max))
        
        # 轉換標註到切片座標
        slice_annotations, kept_objs, total_objs = convert_yolo_to_slice(
            yolo_annotations, image_width, image_height,
            y_min, y_max, x_min, x_max,
            min_area_ratio, min_bbox_size
        )
        
        # 累計統計
        total_kept_objects += kept_objs
        total_original_objects = max(total_original_objects, total_objs)
        
        # 只保留有標註的切片
        if len(slice_annotations) == 0:
            continue
        
        # 生成切片檔名
        slice_filename = f"{image_name}_slice{idx:03d}{image_ext}"
        
        slices_data.append({
            'image': slice_image,
            'annotations': slice_annotations,
            'filename': slice_filename,
            'object_count': len(slice_annotations)
        })
    
    return slices_data, total_original_objects, total_kept_objects

# --- 9. 收集所有切片 ---
def collect_all_slices(input_dirs, min_area_ratio, min_bbox_size):
    """
    收集所有輸入目錄中的圖片並進行切片
    返回所有切片資料的列表和統計資訊
    """
    all_slices = []
    total_images = 0
    total_original_objects = 0
    total_kept_objects = 0
    total_slice_objects = 0  # 切片後的總物件數(可能有重複)
    
    for dir_info in input_dirs:
        image_dir = dir_info['image_dir']
        label_dir = dir_info['label_dir']
        dataset_name = dir_info['name']
        
        print(f"\n正在處理 '{dataset_name}' 資料集...")
        print(f"  圖片來源: {image_dir}")
        print(f"  標註來源: {label_dir}")
        
        # 檢查目錄是否存在
        if not os.path.exists(image_dir):
            print(f"  ⚠️  找不到圖片目錄,跳過")
            continue
        if not os.path.exists(label_dir):
            print(f"  ⚠️  找不到標註目錄,跳過")
            continue
        
        # 取得所有圖片檔案
        image_files = [f for f in os.listdir(image_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        dataset_image_count = len(image_files)
        print(f"  找到 {dataset_image_count} 張圖片")
        
        processed_count = 0
        skipped_count = 0
        dataset_slices = 0
        
        # 處理每張圖片
        for idx, image_file in enumerate(image_files, 1):
            image_path = os.path.join(image_dir, image_file)
            image_name = Path(image_file).stem
            
            # 尋找對應的標註檔案
            label_file = image_name + ".txt"
            label_path = os.path.join(label_dir, label_file)
            
            # 檢查標註檔案是否存在
            if not os.path.exists(label_path):
                if idx % 50 == 0 or idx == dataset_image_count:
                    print(f"  [{idx}/{dataset_image_count}] ⚠️  {image_file} 無標註檔")
                skipped_count += 1
                continue
            
            try:
                # 切片圖片
                slices, orig_objs, kept_objs = slice_single_image(
                    image_path, label_path,
                    SLICE_HEIGHT, SLICE_WIDTH,
                    OVERLAP_HEIGHT_RATIO, OVERLAP_WIDTH_RATIO,
                    min_area_ratio, min_bbox_size
                )
                
                if len(slices) == 0:
                    if idx % 50 == 0 or idx == dataset_image_count:
                        print(f"  [{idx}/{dataset_image_count}] ⚠️  {image_file} 無有效切片")
                    skipped_count += 1
                    continue
                
                # 計算此圖片切片後的總物件數
                slice_obj_count = sum(s['object_count'] for s in slices)
                
                all_slices.extend(slices)
                total_original_objects += orig_objs
                total_kept_objects += kept_objs
                total_slice_objects += slice_obj_count
                processed_count += 1
                dataset_slices += len(slices)
                
                if idx % 50 == 0 or idx == dataset_image_count:
                    print(f"  [{idx}/{dataset_image_count}] ✓ {image_file}: "
                          f"{len(slices)} 切片, {orig_objs}→{kept_objs} 物件")
                    
            except Exception as e:
                print(f"  [{idx}/{dataset_image_count}] ❌ {image_file}: {str(e)}")
                skipped_count += 1
                continue
        
        total_images += processed_count
        print(f"\n  '{dataset_name}' 完成:")
        print(f"    - 成功: {processed_count} 張")
        print(f"    - 跳過: {skipped_count} 張")
        print(f"    - 產生: {dataset_slices} 個切片")
    
    # 顯示總體統計
    print(f"\n{'='*60}")
    print(f"總體統計:")
    print(f"{'='*60}")
    print(f"  處理圖片: {total_images} 張")
    print(f"  產生切片: {len(all_slices)} 個")
    print(f"\n  物件統計:")
    print(f"    - 原始物件總數: {total_original_objects}")
    print(f"    - 至少出現一次: {total_kept_objects} ({total_kept_objects/total_original_objects*100:.1f}%)" if total_original_objects > 0 else "")
    print(f"    - 切片中總數: {total_slice_objects} (含重複)")
    print(f"    - 平均重複率: {total_slice_objects/total_kept_objects:.2f}x" if total_kept_objects > 0 else "")
    
    if total_original_objects > 0:
        lost_objects = total_original_objects - total_kept_objects
        if lost_objects > 0:
            print(f"\n  ⚠️  {lost_objects} 個物件未保留 ({lost_objects/total_original_objects*100:.1f}%)")
            print(f"      (可能因物件太小或可見度不足)")
    
    return all_slices

# --- 10. 儲存切片到指定目錄 ---
def save_slices(slices, output_dir, split_name):
    """
    儲存切片到指定的 train 或 val 目錄
    """
    image_output_dir = os.path.join(output_dir, "images", split_name)
    label_output_dir = os.path.join(output_dir, "labels", split_name)
    
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)
    
    saved_count = 0
    total_objects = 0
    
    for slice_data in slices:
        # 儲存圖片
        image_path = os.path.join(image_output_dir, slice_data['filename'])
        slice_data['image'].save(image_path)
        
        # 儲存標註
        label_filename = Path(slice_data['filename']).stem + ".txt"
        label_path = os.path.join(label_output_dir, label_filename)
        with open(label_path, 'w') as f:
            f.write('\n'.join(slice_data['annotations']))
        
        saved_count += 1
        total_objects += slice_data['object_count']
    
    return saved_count, total_objects

# --- 11. 主程式 ---
if __name__ == "__main__":
    print("=" * 60)
    print("SAHI 風格圖片切片工具")
    print("策略: 最大化保留原始標註資料")
    print("=" * 60)
    
    print(f"\n切片設定:")
    print(f"  切片尺寸: {SLICE_WIDTH}x{SLICE_HEIGHT}")
    print(f"  重疊比例: {OVERLAP_WIDTH_RATIO*100:.0f}% (寬) x {OVERLAP_HEIGHT_RATIO*100:.0f}% (高)")
    print(f"  最小可見面積: {MIN_AREA_RATIO*100:.0f}% (相對原始物件)")
    print(f"  最小邊界框: {MIN_BBOX_SIZE} 像素")
    
    # 定義所有輸入目錄
    input_dirs = [
        {
            'name': 'train',
            'image_dir': os.path.join(BASE_INPUT_DIR, "images", "train"),
            'label_dir': os.path.join(BASE_INPUT_DIR, "labels", "train")
        },
        {
            'name': 'val',
            'image_dir': os.path.join(BASE_INPUT_DIR, "images", "val"),
            'label_dir': os.path.join(BASE_INPUT_DIR, "labels", "val")
        }
    ]
    
    # 步驟 1: 收集所有切片
    print("\n" + "=" * 60)
    print("步驟 1: 切片所有圖片")
    print("=" * 60)
    all_slices = collect_all_slices(input_dirs, MIN_AREA_RATIO, MIN_BBOX_SIZE)
    
    total_slices = len(all_slices)
    
    if total_slices == 0:
        print("\n❌ 沒有產生任何切片,請檢查:")
        print("   1. 輸入目錄路徑是否正確")
        print("   2. 圖片和標註檔案是否存在")
        print("   3. 標註檔案格式是否正確")
        exit(1)
    
    # 步驟 2: 隨機打亂並切分資料集
    print("\n" + "=" * 60)
    print("步驟 2: 打亂並切分資料集")
    print("=" * 60)
    
    random.seed(42)  # 固定隨機種子確保可重現
    random.shuffle(all_slices)
    
    # 計算切分點
    train_count = int(total_slices * TRAIN_RATIO)
    train_slices = all_slices[:train_count]
    val_slices = all_slices[train_count:]
    
    print(f"  訓練集: {len(train_slices)} 個切片 ({TRAIN_RATIO*100:.0f}%)")
    print(f"  驗證集: {len(val_slices)} 個切片 ({(1-TRAIN_RATIO)*100:.0f}%)")
    
    # 步驟 3: 儲存切片
    print("\n" + "=" * 60)
    print("步驟 3: 儲存切片")
    print("=" * 60)
    
    print("  儲存訓練集...")
    train_saved, train_objs = save_slices(train_slices, BASE_OUTPUT_DIR, "train")
    print(f"    ✓ {train_saved} 個切片, {train_objs} 個物件標註")
    
    print("  儲存驗證集...")
    val_saved, val_objs = save_slices(val_slices, BASE_OUTPUT_DIR, "val")
    print(f"    ✓ {val_saved} 個切片, {val_objs} 個物件標註")
    
    # 完成
    print("\n" + "=" * 60)
    print("✅ 所有處理完成!")
    print("=" * 60)
    print(f"\n輸出目錄: {BASE_OUTPUT_DIR}/")
    print(f"  ├── images/")
    print(f"  │   ├── train/ ({train_saved} 張)")
    print(f"  │   └── val/   ({val_saved} 張)")
    print(f"  └── labels/")
    print(f"      ├── train/ ({train_saved} 個, {train_objs} 物件)")
    print(f"      └── val/   ({val_saved} 個, {val_objs} 物件)")
    
    # YAML 配置提示
    print(f"\n" + "=" * 60)
    print("YAML 配置檔範例:")
    print("=" * 60)
    print(f"path: {os.path.abspath(BASE_OUTPUT_DIR)}")
    print(f"train: images/train")
    print(f"val: images/val")
    print(f"names:")
    print(f"  0: your_class_name  # 請替換成你的類別名稱")
    
    print("\n" + "=" * 60)
    print("建議:")
    print("=" * 60)
    print("1. 檢查輸出的切片圖片和標註是否正確")
    print("2. 如果物件保留率太低,可以:")
    print(f"   - 降低 MIN_AREA_RATIO (當前: {MIN_AREA_RATIO})")
    print(f"   - 增加 OVERLAP_RATIO (當前: {OVERLAP_WIDTH_RATIO})")
    print(f"   - 降低 MIN_BBOX_SIZE (當前: {MIN_BBOX_SIZE})")
    print("3. 如果切片數量太多,可以增加 MIN_AREA_RATIO")