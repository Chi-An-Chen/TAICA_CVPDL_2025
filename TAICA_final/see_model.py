import timm
import gc
import torch
from tqdm import tqdm

# 獲取所有 timm 模型列表並過濾掉 test_* 模型
all_models = timm.list_models()
model_list = [m for m in all_models if not m.startswith('test_')]
print(f"總共有 {len(all_models)} 個模型")
print(f"過濾掉 {len(all_models) - len(model_list)} 個 test_* 模型")
print(f"剩餘 {len(model_list)} 個模型待檢查\n")

def count_parameters(model):
    """計算模型的參數量"""
    return sum(p.numel() for p in model.parameters())

def format_params(params):
    """格式化參數數量顯示（以百萬M為單位）"""
    return f"{params / 1e6:.2f}M"

def clear_memory():
    """清理記憶體"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 儲存分類模型的資訊
classification_models = []

print("正在檢查所有模型...\n")

# 使用 tqdm 顯示進度條
for model_name in tqdm(model_list, desc="檢查模型", unit="模型"):
    try:
        # 創建模型（不載入預訓練權重以加快速度）
        model = timm.create_model(model_name, pretrained=False)
        
        # 檢查是否有分類頭（classifier）
        has_classifier = hasattr(model, 'num_classes') and hasattr(model, 'get_classifier')
        
        if has_classifier:
            num_params = count_parameters(model)
            num_classes = model.num_classes
            
            classification_models.append({
                'name': model_name,
                'params': num_params,
                'num_classes': num_classes
            })
        
        # 刪除模型釋放記憶體
        del model
        
        # 每處理50個模型後清理一次記憶體
        if len(classification_models) % 50 == 0:
            clear_memory()
            
    except Exception as e:
        # 靜默處理錯誤，但可以記錄到 tqdm 的 postfix
        pass
    except MemoryError:
        # 如果發生記憶體錯誤，立即清理
        tqdm.write(f"⚠️ 記憶體不足於 {model_name}，正在清理...")
        clear_memory()
        continue

# 最後再清理一次記憶體
clear_memory()

print(f"\n檢查完成！\n")

# 顯示摘要
print("=" * 80)
print(f"找到 {len(classification_models)} 個分類模型")
print("=" * 80)

if classification_models:
    # 按參數量排序並取前100名
    classification_models.sort(key=lambda x: x['params'])
    top_100 = classification_models[:100]
    
    # 準備輸出內容
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append(f"找到 {len(classification_models)} 個分類模型")
    output_lines.append("=" * 80)
    output_lines.append("")
    output_lines.append("參數量最小的前100名分類模型：")
    output_lines.append("=" * 80)
    
    header = f"{'排名':<6} {'模型名稱':<45} {'參數量':<15} {'分類數':<10}"
    output_lines.append(header)
    output_lines.append("-" * 80)
    
    # 顯示並記錄前100名
    print("\n參數量最小的前100名分類模型：")
    print("=" * 80)
    print(header)
    print("-" * 80)
    
    for rank, model_info in enumerate(top_100, 1):
        line = f"{rank:<6} {model_info['name']:<45} {format_params(model_info['params']):<15} {model_info['num_classes']:<10}"
        print(line)
        output_lines.append(line)
    
    output_lines.append("=" * 80)
    print("=" * 80)
    
    # 將結果寫入 txt 檔案
    output_file = "timm_top100_models.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"\n✓ 結果已儲存至 {output_file}")
    
else:
    print("未找到任何分類模型")