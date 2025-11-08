import os
import re
import cv2
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel


def yolo_sahi_sliced_inference(model_path, test_dir, output_csv="predictions_sahi.csv", 
                               conf_threshold=0.5, save_images=True, 
                               output_img_dir="prediction_results_sahi",
                               slice_height=256, slice_width=256,
                               overlap_height_ratio=0.2, overlap_width_ratio=0.2):
    """
    使用SAHI進行切片推理，將YOLO預測結果轉換為指定格式，並儲存有檢測結果的圖片
    
    Args:
        model_path: 訓練好的模型路徑
        test_dir: 測試圖片目錄
        output_csv: 輸出CSV檔案名稱
        conf_threshold: 信心度閾值
        save_images: 是否儲存有檢測結果的圖片
        output_img_dir: 輸出圖片的目錄
        slice_height: 切片高度
        slice_width: 切片寬度
        overlap_height_ratio: 高度重疊比例
        overlap_width_ratio: 寬度重疊比例
    """
    
    # 設定裝置
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"使用裝置: {device}")
    
    # 載入原始YOLO模型（用於獲取類別名稱）
    yolo_model = YOLO(model_path)
    yolo_model.to(device)
    
    # 為SAHI載入模型 - 使用官方推薦方式
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=conf_threshold,
        device=device
    )
    
    # 準備結果列表
    results_data = []
    
    # 建立輸出目錄
    if save_images:
        os.makedirs(output_img_dir, exist_ok=True)
        print(f"檢測結果圖片將保存到: {output_img_dir}")
    
    # 取得所有測試圖片
    test_images = sorted(Path(test_dir).glob("*.png"))
    
    print(f"開始處理 {len(test_images)} 張測試圖片...")
    print(f"切片設定 - 高: {slice_height}, 寬: {slice_width}, 高重疊: {overlap_height_ratio}, 寬重疊: {overlap_width_ratio}")
    
    # 統計變數
    images_with_detections = 0
    total_detections = 0
    
    for img_path in tqdm(test_images, desc="SAHI Predicting", unit="img"):
        # 取得圖片ID (去掉副檔名)
        image_id = re.sub(r'\D', '', img_path.stem)
        image_id = str(int(image_id))
        
        # 讀取原始圖片
        original_img = cv2.imread(str(img_path))
        img_height, img_width = original_img.shape[:2]
        
        # 使用SAHI進行切片推理 - 修改這裡：使用 image 參數而非 image_path
        try:
            result = get_sliced_prediction(
                image=str(img_path),  # 改用 image 參數
                detection_model=detection_model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio,
            )
        except Exception as e:
            print(f"錯誤處理 {img_path.name}: {e}")
            results_data.append({
                'Image_ID': image_id,
                'PredictionString': "0.0 0.0 0.0 0.0 0.0 0"
            })
            continue
        
        # 處理預測結果
        prediction_string = ""
        has_detections = False
        
        # 提取檢測結果
        object_prediction_list = result.object_prediction_list
        
        if object_prediction_list is not None and len(object_prediction_list) > 0:
            has_detections = True
            images_with_detections += 1
            total_detections += len(object_prediction_list)
            
            for obj_pred in object_prediction_list:
                # 取得信心度和類別
                conf = obj_pred.score.value
                cls = int(obj_pred.category.id) if obj_pred.category.id is not None else 0
                
                # 取得邊界框座標 (SAHI返回的是 [x1, y1, x2, y2] 格式)
                bbox = obj_pred.bbox.minx, obj_pred.bbox.miny, obj_pred.bbox.maxx, obj_pred.bbox.maxy
                x1, y1, x2, y2 = bbox
                
                # 轉換為指定格式 (left, top, width, height)
                bb_left = x1
                bb_top = y1
                bb_width = x2 - x1
                bb_height = y2 - y1
                
                # 組合預測字串
                pred_part = f"{conf:.4f} {bb_left:.1f} {bb_top:.1f} {bb_width:.1f} {bb_height:.1f} {cls}"
                
                if prediction_string:
                    prediction_string += f" {pred_part}"
                else:
                    prediction_string = pred_part
                
                # 在圖片上畫邊界框（如果需要儲存圖片）
                if save_images:
                    # 畫邊界框
                    cv2.rectangle(original_img, 
                                (int(x1), int(y1)), 
                                (int(x2), int(y2)), 
                                (0, 255, 0), 2)
                    
                    # 添加信心度和類別標籤
                    class_name = yolo_model.names.get(cls, f"class_{cls}")
                    label = f"{class_name}: {conf:.3f}"
                    
                    # 計算標籤位置
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    label_y = max(int(y1), label_size[1] + 10)
                    
                    # 畫標籤背景
                    cv2.rectangle(original_img,
                                (int(x1), label_y - label_size[1] - 10),
                                (int(x1) + label_size[0], label_y),
                                (0, 255, 0), -1)
                    
                    # 畫標籤文字
                    cv2.putText(original_img, label,
                              (int(x1), label_y - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                              (0, 0, 0), 1)
        
        # 如果沒有預測結果
        if not prediction_string:
            prediction_string = "0.0 0.0 0.0 0.0 0.0 0"
        
        # 儲存有檢測結果的圖片
        if save_images and has_detections:
            output_img_path = os.path.join(output_img_dir, f"{img_path.stem}_detected_sahi.jpg")
            cv2.imwrite(output_img_path, original_img)
        
        # 添加到結果列表
        results_data.append({
            'Image_ID': image_id,
            'PredictionString': prediction_string
        })
    
    # 建立DataFrame並保存
    df = pd.DataFrame(results_data)
    df.to_csv(output_csv, index=False)
    
    # 統計報告
    print(f"\n=== SAHI切片推理統計 ===")
    print(f"總圖片數: {len(results_data)}")
    print(f"有檢測結果的圖片: {images_with_detections}")
    print(f"總檢測物件數: {total_detections}")
    print(f"平均每張圖片檢測物件數: {total_detections/len(results_data):.2f}")
    if save_images:
        print(f"檢測結果圖片已保存到: {output_img_dir}")
    print(f"預測結果CSV已保存到: {output_csv}")
    
    return df


# === 使用範例 ===
if __name__ == "__main__":
    model_path = "./results_finetune/2025-10-19/weights/best.pt"
    test_directory = "../../dataset/CVPDL_hw2/test"
    
    # 使用SAHI進行切片推理
    predictions_df = yolo_sahi_sliced_inference(
        model_path=model_path,
        test_dir=test_directory,
        output_csv="predictions_finetune_sahi.csv",
        conf_threshold=0.20,
        save_images=True,
        output_img_dir="predictio_finetune_sahi",
        slice_height=540,
        slice_width=960,
        overlap_height_ratio=0.25,
        overlap_width_ratio=0.25
    )
    
    print("\n預測結果預覽:")
    print(predictions_df.head())