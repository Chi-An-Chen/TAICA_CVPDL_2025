import os
import re
import cv2
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO


def yolo_to_custom_format_with_visualization(model_path, test_dir, output_csv="predictions.csv", 
                                           conf_threshold=0.5, save_images=True, 
                                           output_img_dir="prediction_results"):
    """
    將YOLO預測結果轉換為指定格式，並同時儲存有檢測結果的圖片 
    
    Args:
        model_path: 訓練好的模型路徑
        test_dir: 測試圖片目錄
        output_csv: 輸出CSV檔案名稱
        conf_threshold: 信心度閾值
        save_images: 是否儲存有檢測結果的圖片
        output_img_dir: 輸出圖片的目錄
    """
    
    # 載入訓練好的模型
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(model_path)
    model.to(device)

    # 準備結果列表
    results_data = []
    
    # 建立輸出目錄
    if save_images:
        os.makedirs(output_img_dir, exist_ok=True)
        print(f"檢測結果圖片將保存到: {output_img_dir}")
    
    # 取得所有測試圖片
    test_images = sorted(Path(test_dir).glob("*.png"))
    
    print(f"開始處理 {len(test_images)} 張測試圖片...")
    
    # 統計變數
    images_with_detections = 0
    total_detections = 0
    
    for img_path in tqdm(test_images, desc="Predicting", unit="img"):
        # 取得圖片ID (去掉副檔名)
        image_id = re.sub(r'\D', '', img_path.stem)  # 移除所有非數字字元
        image_id = str(int(image_id))  # 轉成整數再轉回字串，去掉前導零
        # image_id = str(int(img_path.stem))
        
        # 讀取原始圖片用於可視化
        if save_images:
            original_img = cv2.imread(str(img_path))
            img_height, img_width = original_img.shape[:2]
        
        # 進行預測
        results = model.predict(
            source=str(img_path),
            conf=conf_threshold,
            verbose=False,
            save=False
        )
        
        # 處理預測結果
        prediction_string = ""
        has_detections = False
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            detection_count = len(boxes)
            
            if detection_count > 0:
                has_detections = True
                images_with_detections += 1
                total_detections += detection_count
            
            for i in range(detection_count):
                # 取得預測資訊
                conf = float(boxes.conf[i])
                cls = int(boxes.cls[i])
                
                # 取得邊界框座標 (xyxy格式)
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                
                # 轉換為您要的格式 (left, top, width, height)
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
                                (0, 255, 0), 2)  # 綠色邊界框
                    
                    # 添加信心度和類別標籤
                    class_name = model.names[cls] if cls in model.names else f"class_{cls}"
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
        
        # 如果沒有預測結果，設為空字串
        if not prediction_string:
            # prediction_string = ""
            prediction_string = "0.0 0.0 0.0 0.0 0.0 0"
    
        
        # 儲存有檢測結果的圖片
        if save_images and has_detections:
            output_img_path = os.path.join(output_img_dir, f"{img_path.stem}_detected.jpg")
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
    print(f"\n=== 預測統計 ===")
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
    model_path = "./results_finetune/2025-10-25/weights/best.pt"
    test_directory = "../../dataset/CVPDL_hw2/test"
    
    # 方法1: 儲存所有有檢測結果的圖片
    predictions_df = yolo_to_custom_format_with_visualization(
        model_path=model_path,
        test_dir=test_directory,
        output_csv="predictione_finetune_0.05.csv",
        conf_threshold=0.05,
        save_images=True,
        output_img_dir="prediction_finetune_0.05"
    )
    
    print("\n預測結果預覽:")
    print(predictions_df.head())