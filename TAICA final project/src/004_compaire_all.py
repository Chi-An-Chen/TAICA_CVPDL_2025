import json
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def count_parameters(model_path):
    """計算模型參數量"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        total_params = sum(p.numel() for p in state_dict.values())
        return total_params
    except Exception as e:
        print(f"無法讀取 {model_path}: {e}")
        return None

def load_results(result_folder='./results'):
    """讀取所有JSON結果檔案"""
    results = {}
    result_path = Path(result_folder)
    
    # 檢查資料夾是否存在
    if not result_path.exists():
        print(f"警告: 找不到資料夾 '{result_folder}'")
        return results
    
    json_files = list(result_path.glob('*_strategy_comparison.json'))
    print(f"找到 {len(json_files)} 個JSON檔案")
    
    for json_file in json_files:
        print(f"正在讀取: {json_file.name}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                model_name = data['model_name']
                results[model_name] = data
                print(f"  - 成功讀取模型: {model_name}")
        except Exception as e:
            print(f"  - 讀取失敗: {e}")
    
    return results

def get_model_parameters(model_name, checkpoints_base='./'):
    """獲取模型參數量"""
    # 根據模型名稱找到對應的checkpoint資料夾
    checkpoint_folders = { 
        'dla46x_c': 'checkpoints_comparison_dla',
        # 'convnextv2_large': 'checkpoints_comparison_large',
        'mnasnet_small': 'checkpoints_comparison_mnasnet',
        'mobilenetv3_small_050': 'checkpoints_comparison_mobilenetv3',
        'mobilevit_xxs': 'checkpoints_comparison_mobilevit',
        'mobilevitv2_050': 'checkpoints_comparison_mobilevit_v2',
        'convnextv2_atto': 'checkpoints_comparison_atto',
        'starnet_s1': 'checkpoints_comparison_starnet',
        'edgenext_xx_small': 'checkpoints_comparison_edgenext',
    }
    
    folder = checkpoint_folders.get(model_name)
    if not folder:
        print(f"找不到模型 {model_name} 的checkpoint資料夾")
        return None
    
    # 使用finetune模型來計算參數(scratch和finetune模型架構相同)
    model_path = Path(checkpoints_base) / folder / 'best_finetune_model.pth'
    
    if not model_path.exists():
        print(f"找不到模型檔案: {model_path}")
        return None
    
    return count_parameters(model_path)

def create_comparison_plots(results):
    """創建比較散點圖"""
    strategies = ['scratch', 'finetune', 'pretrained_only']
    strategy_names = {
        'scratch': '從頭訓練',
        'finetune': '微調',
        'pretrained_only': '僅預訓練權重'
    }
    
    metrics = ['accuracy', 'f1_macro', 'precision_weighted']
    metric_names = {
        'accuracy': '準確率 (Accuracy)',
        'f1_macro': 'F1分數 (Macro)',
        'precision_weighted': '精確率 (Weighted)'
    }
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('模型訓練策略比較：性能指標 vs 參數量', fontsize=16, y=0.995)
    
    colors = {'scratch': '#FF6B6B', 'finetune': '#4ECDC4', 'pretrained_only': '#95E1D3'}
    markers = {'scratch': 'o', 'finetune': 's', 'pretrained_only': '^'}
    
    for strategy_idx, strategy in enumerate(strategies):
        for metric_idx, metric in enumerate(metrics):
            ax = axes[strategy_idx, metric_idx]
            
            model_names = []
            params_list = []
            metric_values = []
            
            for model_name, data in results.items():
                params = get_model_parameters(model_name)
                if params is None:
                    continue
                
                test_results = data['test_results'].get(strategy, {})
                if not test_results or test_results.get(metric) is None:
                    continue
                
                model_names.append(model_name)
                params_list.append(params / 1e6)  # 轉換為百萬
                metric_values.append(test_results[metric])
            
            # 繪製散點圖
            ax.scatter(params_list, metric_values, 
                      c=colors[strategy], marker=markers[strategy],
                      s=150, alpha=0.7, edgecolors='black', linewidth=1.5)
            
            # 標註模型名稱
            for i, name in enumerate(model_names):
                ax.annotate(name, (params_list[i], metric_values[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
            
            # 設定標籤和標題
            if strategy_idx == 0:
                ax.set_title(metric_names[metric], fontsize=12, fontweight='bold')
            if metric_idx == 0:
                ax.set_ylabel(f"{strategy_names[strategy]}\n分數", fontsize=11)
            if strategy_idx == 2:
                ax.set_xlabel('參數量 (百萬)', fontsize=10)
            
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig('model_comparison_plots.png', dpi=300, bbox_inches='tight')
    print("散點圖已儲存為 'model_comparison_plots.png'")
    plt.show()

def create_detailed_tables(results):
    """創建詳細的結果表格"""
    strategies = ['scratch', 'finetune', 'pretrained_only']
    strategy_names = {
        'scratch': '從頭訓練',
        'finetune': '微調',
        'pretrained_only': '僅預訓練權重'
    }
    
    for strategy in strategies:
        print(f"\n{'='*100}")
        print(f"{strategy_names[strategy]}策略 - 詳細測試結果")
        print(f"{'='*100}\n")
        
        table_data = []
        
        for model_name, data in results.items():
            params = get_model_parameters(model_name)
            test_results = data['test_results'].get(strategy, {})
            
            if not test_results:
                continue
            
            row = {
                '模型名稱': model_name,
                '參數量 (M)': f"{params/1e6:.2f}" if params else 'N/A',
                '準確率': f"{test_results.get('accuracy', 0):.4f}",
                'F1 (Weighted)': f"{test_results.get('f1_weighted', 0):.4f}",
                'F1 (Macro)': f"{test_results.get('f1_macro', 0):.4f}",
                '精確率 (Weighted)': f"{test_results.get('precision_weighted', 0):.4f}",
                '召回率 (Weighted)': f"{test_results.get('recall_weighted', 0):.4f}"
            }
            table_data.append(row)
        
        # 創建DataFrame並排序
        df = pd.DataFrame(table_data)
        if not df.empty:
            df = df.sort_values('準確率', ascending=False)
            print(df.to_string(index=False))
            
            # 儲存為CSV
            csv_filename = f'table_{strategy}.csv'
            df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            print(f"\n表格已儲存為 '{csv_filename}'")
        else:
            print(f"沒有 {strategy_names[strategy]} 的測試結果")

def create_summary_comparison(results):
    """創建總結比較表"""
    print(f"\n{'='*120}")
    print("所有模型與策略的總結比較")
    print(f"{'='*120}\n")
    
    table_data = []
    
    for model_name, data in results.items():
        params = get_model_parameters(model_name)
        
        for strategy in ['scratch', 'finetune', 'pretrained_only']:
            test_results = data['test_results'].get(strategy, {})
            if not test_results:
                continue
            
            row = {
                '模型名稱': model_name,
                '訓練策略': {'scratch': '從頭訓練', 'finetune': '微調', 'pretrained_only': '僅預訓練'}[strategy],
                '參數量 (M)': f"{params/1e6:.2f}" if params else 'N/A',
                '準確率': f"{test_results.get('accuracy', 0):.4f}",
                'F1 (Macro)': f"{test_results.get('f1_macro', 0):.4f}",
                '精確率': f"{test_results.get('precision_weighted', 0):.4f}"
            }
            table_data.append(row)
    
    if not table_data:
        print("沒有可用的測試結果")
        return
    
    df = pd.DataFrame(table_data)
    df = df.sort_values(['模型名稱', '準確率'], ascending=[True, False])
    print(df.to_string(index=False))
    
    # 儲存總結表
    df.to_csv('summary_comparison.csv', index=False, encoding='utf-8-sig')
    print(f"\n總結表已儲存為 'summary_comparison.csv'")

def main():
    print("開始分析模型比較結果...")
    
    # 讀取結果
    results = load_results('results')
    
    if not results:
        print("\n錯誤: 沒有找到任何結果檔案!")
        return
    
    print(f"\n成功讀取 {len(results)} 個模型的結果\n")
    
    # 創建散點圖
    print("生成比較散點圖...")
    create_comparison_plots(results)
    
    # 創建詳細表格
    print("\n生成詳細結果表格...")
    create_detailed_tables(results)
    
    # 創建總結比較
    create_summary_comparison(results)
    
    print("\n\n分析完成!")

if __name__ == "__main__":
    main()