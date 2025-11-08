import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def optimize_threshold_for_map(model_path, data_yaml_path, output_dir="threshold_analysis"):
    """
    針對 mAP@0.5:0.95 優化信心度閾值
    
    Args:
        model_path: 訓練好的模型路徑
        data_yaml_path: 驗證數據的yaml配置文件
        output_dir: 結果輸出目錄
    """
    
    # 載入模型
    model = YOLO(model_path)
    
    # 測試的閾值範圍
    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    
    results = []
    best_threshold = 0.25
    best_map50_95 = 0
    best_map50 = 0
    
    print("開始測試不同信心度閾值...")
    
    for threshold in thresholds:
        # 使用當前閾值進行驗證
        metrics = model.val(
            data=data_yaml_path,
            conf=threshold,
            verbose=False,
            plots=False,
            save=False,
            split='val',
            device=device
        )
        
        # 獲取關鍵指標
        map50_95 = float(metrics.box.map)      # mAP@0.5:0.95
        map50 = float(metrics.box.map50)       # mAP@0.5
        precision = float(metrics.box.mp)      # mean Precision
        recall = float(metrics.box.mr)         # mean Recall
        
        # 記錄結果
        result = {
            'threshold': threshold,
            'map50_95': map50_95,
            'map50': map50,
            'precision': precision,
            'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        }
        results.append(result)
        
        # 更新最佳閾值（以mAP@0.5:0.95為準）
        if map50_95 > best_map50_95:
            best_map50_95 = map50_95
            best_map50 = map50
            best_threshold = threshold
    
    print("\n" + "="*65)
    print(f"最佳閾值: {best_threshold}")
    print(f"最佳 mAP@0.5:0.95: {best_map50_95:.4f}")
    print(f"對應 mAP@0.5: {best_map50:.4f}")
    
    # 繪製分析圖表
    plot_threshold_metrics(results, output_dir)
    
    # 保存詳細結果
    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/threshold_analysis.csv", index=False)
    
    return best_threshold, results

def plot_threshold_metrics(results, output_dir="threshold_analysis"):
    """繪製閾值分析圖表"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    thresholds = [r['threshold'] for r in results]
    map50_95 = [r['map50_95'] for r in results]
    map50 = [r['map50'] for r in results]
    precision = [r['precision'] for r in results]
    recall = [r['recall'] for r in results]
    f1_scores = [r['f1'] for r in results]
    
    plt.figure(figsize=(15, 10))
    
    # mAP 圖表
    plt.subplot(2, 3, 1)
    plt.plot(thresholds, map50_95, 'b-o', label='mAP@0.5:0.95', linewidth=2)
    plt.plot(thresholds, map50, 'r-o', label='mAP@0.5', linewidth=2)
    plt.xlabel('Confidence Threshold')
    plt.ylabel('mAP')
    plt.title('mAP vs Confidence Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Precision & Recall
    plt.subplot(2, 3, 2)
    plt.plot(thresholds, precision, 'g-o', label='Precision', linewidth=2)
    plt.plot(thresholds, recall, 'orange', marker='o', label='Recall', linewidth=2)
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Score')
    plt.title('Precision & Recall vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # F1 Score
    plt.subplot(2, 3, 3)
    plt.plot(thresholds, f1_scores, 'm-o', linewidth=2)
    plt.xlabel('Confidence Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Threshold')
    plt.grid(True, alpha=0.3)
    
    # mAP@0.5:0.95 重點圖
    plt.subplot(2, 3, 4)
    best_idx = map50_95.index(max(map50_95))
    plt.plot(thresholds, map50_95, 'b-o', linewidth=2)
    plt.plot(thresholds[best_idx], map50_95[best_idx], 'ro', markersize=10, label=f'Best: {thresholds[best_idx]}')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('mAP@0.5:0.95')
    plt.title('mAP@0.5:0.95 (Main Metric)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Precision-Recall 曲線
    plt.subplot(2, 3, 5)
    plt.plot(recall, precision, 'g-o', linewidth=2)
    for i, threshold in enumerate(thresholds):
        if i % 2 == 0:  # 每隔一個點標註
            plt.annotate(f'{threshold:.2f}', (recall[i], precision[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    
    # 綜合指標對比
    plt.subplot(2, 3, 6)
    x = range(len(thresholds))
    width = 0.35
    plt.bar([i - width/2 for i in x], map50_95, width, label='mAP@0.5:0.95', alpha=0.8)
    plt.bar([i + width/2 for i in x], f1_scores, width, label='F1 Score', alpha=0.8)
    plt.xlabel('Threshold Index')
    plt.ylabel('Score')
    plt.title('mAP vs F1 Comparison')
    plt.xticks(x, [f'{t:.2f}' for t in thresholds], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/threshold_analysis.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    # 設定路徑
    model_path = "./results_finetune/2025-10-19/weights/best.pt"
    data_yaml_path = "data.yaml"
    
    best_threshold, results = optimize_threshold_for_map(
        model_path=model_path,
        data_yaml_path=data_yaml_path,
        output_dir="threshold_analysis_finetune"
    )
    
    print(f"\n 建議使用的信心度閾值: {best_threshold:.3f}")
