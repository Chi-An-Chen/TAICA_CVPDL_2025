"""
Author: Chi-An Chen
Date: 2025-12-05
Version: 2.0
Description: 使用 pytorch-grad-cam 函式庫比較 ConvNeXt V2 Large 和 MobileViT XXS 的 GradCAM
             隨機選取 10 張測試圖片，生成並保存兩個模型的 GradCAM 熱圖對比
             
Dependencies: pip install grad-cam
"""
import os
import json
import random
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from utils.dataloader import CitrusDataset, get_transforms


def get_target_layers(model, model_name):
    """
    獲取目標層用於 GradCAM
    
    Args:
        model: 模型
        model_name: 模型名稱
    
    Returns:
        target_layers: 目標層列表
    """
    if 'convnext' in model_name.lower():
        # ConvNeXt: 使用最後一個 stage 的最後一個 block
        target_layers = [model.stages[-1].blocks[-1].norm]
    elif 'mobilevit' in model_name.lower():
        # MobileViT: 使用最後一個 stage 的最後一層
        # MobileViT 結構較複雜，選擇最後的 conv layer
        target_layers = [model.stages[-1][-1]]
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")
    
    return target_layers


def load_model(checkpoint_path, model_name, num_classes, device):
    """載入訓練好的模型"""
    # 創建模型
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    
    # 載入權重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ 成功載入模型: {model_name}")
    print(f"  Checkpoint: {checkpoint_path}")
    if 'val_acc' in checkpoint:
        print(f"  Val Acc: {checkpoint['val_acc']:.4f}")
    if 'val_f1' in checkpoint:
        print(f"  Val F1: {checkpoint['val_f1']:.4f}")
    
    return model, checkpoint.get('class_names', None)


def generate_gradcam(model, target_layers, input_tensor, target_class=None):
    """
    使用 pytorch-grad-cam 生成 GradCAM
    
    Args:
        model: 模型
        target_layers: 目標層列表
        input_tensor: 輸入圖片 tensor (1, C, H, W)
        target_class: 目標類別，None 則使用預測類別
    
    Returns:
        cam: GradCAM 熱圖 (H, W)
        pred_class: 預測類別
        pred_prob: 預測概率
    """
    # 創建 GradCAM 對象
    cam_algorithm = GradCAM(model=model, target_layers=target_layers)
    
    # 獲取預測
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        pred_prob = probs[0, pred_class].item()
    
    # 如果沒有指定目標類別，使用預測類別
    if target_class is None:
        target_class = pred_class
    
    # 生成 GradCAM
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam_algorithm(input_tensor=input_tensor, targets=targets)
    
    # 取第一張圖片的 CAM
    cam = grayscale_cam[0, :]
    
    return cam, pred_class, pred_prob


def visualize_gradcam_comparison(original_img, input_tensor, cam1, cam2, 
                                  model1_name, model2_name,
                                  pred1, pred2, prob1, prob2, 
                                  true_label, class_name, save_path):
    """
    視覺化兩個模型的 GradCAM 對比
    
    Args:
        original_img: 原始圖片 PIL Image
        input_tensor: 輸入 tensor (用於反標準化)
        cam1, cam2: 兩個模型的 GradCAM 熱圖
        model1_name, model2_name: 模型名稱
        pred1, pred2: 預測類別
        prob1, prob2: 預測概率
        true_label: 真實標籤
        class_name: 真實類別名稱
        save_path: 保存路徑
    """
    # 將 PIL Image 轉換為 numpy array (0-1 範圍)
    img_array = np.array(original_img.resize((224, 224))) / 255.0
    
    # 使用 show_cam_on_image 生成疊加圖
    cam1_image = show_cam_on_image(img_array, cam1, use_rgb=True)
    cam2_image = show_cam_on_image(img_array, cam2, use_rgb=True)
    
    # 創建對比圖
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 原始圖片
    axes[0].imshow(original_img)
    axes[0].set_title(f'Original Image\nTrue Label: {class_name} (Class {true_label})', 
                      fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # 模型 1 的 GradCAM
    axes[1].imshow(cam1_image)
    correct1 = '✓' if pred1 == true_label else '✗'
    color1 = 'green' if pred1 == true_label else 'red'
    axes[1].set_title(
        f'{model1_name}\nPred: Class {pred1} (prob: {prob1:.3f}) {correct1}',
        fontsize=12, fontweight='bold', color=color1
    )
    axes[1].axis('off')
    
    # 模型 2 的 GradCAM
    axes[2].imshow(cam2_image)
    correct2 = '✓' if pred2 == true_label else '✗'
    color2 = 'green' if pred2 == true_label else 'red'
    axes[2].set_title(
        f'{model2_name}\nPred: Class {pred2} (prob: {prob2:.3f}) {correct2}',
        fontsize=12, fontweight='bold', color=color2
    )
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """主函數"""
    # 配置
    CONFIG = {
        'model1': {
            'name': 'convnextv2_atto',
            'checkpoint': './checkpoints_comparison_atto/best_finetune_model.pth',
            'display_name': 'ConvNeXt V2 Atto'
        },
        'model2': {
            'name': 'mobilevit_xxs',
            'checkpoint': './checkpoints_comparison_mobilevit/best_finetune_model.pth',
            'display_name': 'MobileViT XXS'
        },
        'test_txt': './TAICA期末資料集/test.txt',
        'root_dir': './TAICA期末資料集',
        'image_size': 224,
        'num_samples': 10,
        'output_dir': './gradcam_comparison',
        'seed': 666
    }
    
    # 設置隨機種子
    random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CONFIG['seed'])
    
    # 設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # 創建輸出目錄
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # 準備數據
    print("準備測試數據集...")
    _, val_transform = get_transforms(CONFIG['image_size'])
    test_dataset = CitrusDataset(
        CONFIG['test_txt'],
        CONFIG['root_dir'],
        transform=val_transform
    )
    
    print(f"測試樣本數: {len(test_dataset)}")
    print(f"類別數量: {len(test_dataset.classes)}")
    print(f"類別列表: {test_dataset.classes}\n")
    
    # 載入兩個模型
    print("="*80)
    print("載入模型")
    print("="*80)
    
    model1, class_names = load_model(
        CONFIG['model1']['checkpoint'],
        CONFIG['model1']['name'],
        len(test_dataset.classes),
        device
    )
    
    print()
    
    model2, _ = load_model(
        CONFIG['model2']['checkpoint'],
        CONFIG['model2']['name'],
        len(test_dataset.classes),
        device
    )
    
    # 獲取目標層
    print("\n" + "="*80)
    print("設置 GradCAM 目標層")
    print("="*80)
    
    target_layers1 = get_target_layers(model1, CONFIG['model1']['name'])
    target_layers2 = get_target_layers(model2, CONFIG['model2']['name'])
    
    print(f"✓ Model 1 目標層: {target_layers1[0]}")
    print(f"✓ Model 2 目標層: {target_layers2[0]}")
    
    # 隨機選擇樣本
    print(f"\n隨機選擇 {CONFIG['num_samples']} 個測試樣本...")
    random_indices = random.sample(range(len(test_dataset)), CONFIG['num_samples'])
    print(f"選中的索引: {random_indices}")
    
    # 讀取測試集圖片路徑
    with open(CONFIG['test_txt'], 'r', encoding='utf-8') as f:
        image_paths = [line.strip().split()[0] for line in f.readlines()]
    
    # 處理每個樣本
    print("\n" + "="*80)
    print("生成 GradCAM 對比圖")
    print("="*80 + "\n")
    
    results = []
    
    for idx, sample_idx in enumerate(random_indices, 1):
        print(f"處理樣本 {idx}/{CONFIG['num_samples']} (Dataset Index: {sample_idx})...")
        
        # 獲取樣本
        img_tensor, label = test_dataset[sample_idx]
        img_path = os.path.join(CONFIG['root_dir'], image_paths[sample_idx])
        class_name = test_dataset.classes[label]
        
        # 載入原始圖片用於顯示
        original_img = Image.open(img_path).convert('RGB')
        
        # 準備模型輸入
        input_tensor = img_tensor.unsqueeze(0).to(device)
        
        # 生成 GradCAM
        try:
            cam1, pred1, prob1 = generate_gradcam(model1, target_layers1, input_tensor)
            cam2, pred2, prob2 = generate_gradcam(model2, target_layers2, input_tensor)
            
            # 視覺化並保存
            save_path = os.path.join(
                CONFIG['output_dir'],
                f'gradcam_comparison_{idx:02d}_class{label}_{class_name}.png'
            )
            
            visualize_gradcam_comparison(
                original_img, input_tensor, cam1, cam2,
                CONFIG['model1']['display_name'],
                CONFIG['model2']['display_name'],
                pred1, pred2, prob1, prob2,
                label, class_name, save_path
            )
            
            # 記錄結果
            result = {
                'sample_id': idx,
                'dataset_index': sample_idx,
                'image_path': img_path,
                'true_label': int(label),
                'true_class': class_name,
                'model1': {
                    'name': CONFIG['model1']['display_name'],
                    'prediction': int(pred1),
                    'probability': float(prob1),
                    'correct': bool(pred1 == label)
                },
                'model2': {
                    'name': CONFIG['model2']['display_name'],
                    'prediction': int(pred2),
                    'probability': float(prob2),
                    'correct': bool(pred2 == label)
                }
            }
            results.append(result)
            
            print(f"  真實標籤: {class_name} (Class {label})")
            print(f"  {CONFIG['model1']['display_name']}: Class {pred1} (prob: {prob1:.3f}) {'✓' if pred1 == label else '✗'}")
            print(f"  {CONFIG['model2']['display_name']}: Class {pred2} (prob: {prob2:.3f}) {'✓' if pred2 == label else '✗'}")
            print(f"  ✓ 保存至: {save_path}\n")
            
        except Exception as e:
            print(f"  ✗ 處理失敗: {str(e)}\n")
            continue
    
    # 統計結果
    print("="*80)
    print("統計結果")
    print("="*80 + "\n")
    
    if results:
        model1_correct = sum(1 for r in results if r['model1']['correct'])
        model2_correct = sum(1 for r in results if r['model2']['correct'])
        both_correct = sum(1 for r in results if r['model1']['correct'] and r['model2']['correct'])
        both_wrong = sum(1 for r in results if not r['model1']['correct'] and not r['model2']['correct'])
        
        print(f"{CONFIG['model1']['display_name']}:")
        print(f"  正確: {model1_correct}/{len(results)} ({model1_correct/len(results)*100:.1f}%)")
        print()
        print(f"{CONFIG['model2']['display_name']}:")
        print(f"  正確: {model2_correct}/{len(results)} ({model2_correct/len(results)*100:.1f}%)")
        print()
        print(f"兩者都正確: {both_correct}/{len(results)} ({both_correct/len(results)*100:.1f}%)")
        print(f"兩者都錯誤: {both_wrong}/{len(results)} ({both_wrong/len(results)*100:.1f}%)")
        
        # 保存詳細結果
        results_file = os.path.join(CONFIG['output_dir'], 'gradcam_comparison_results.json')
        summary = {
            'config': CONFIG,
            'statistics': {
                'total_samples': len(results),
                'model1': {
                    'name': CONFIG['model1']['display_name'],
                    'correct': model1_correct,
                    'accuracy': f"{model1_correct/len(results)*100:.2f}%"
                },
                'model2': {
                    'name': CONFIG['model2']['display_name'],
                    'correct': model2_correct,
                    'accuracy': f"{model2_correct/len(results)*100:.2f}%"
                },
                'both_correct': both_correct,
                'both_wrong': both_wrong
            },
            'samples': results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        
        print(f"\n✓ 詳細結果已保存至: {results_file}")
        print(f"✓ GradCAM 對比圖已保存至: {CONFIG['output_dir']}/")
    else:
        print("✗ 沒有成功處理的樣本")
    
    print(f"\n所有處理完成!")


if __name__ == "__main__":
    main()