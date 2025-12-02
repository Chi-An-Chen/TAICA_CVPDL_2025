"""
Author: Chi-An Chen
Date: 2025-12-02
Version: 2.1
Description:
"""

import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision.models import (
    convnext_large, ConvNeXt_Large_Weights,
    efficientnet_v2_l, EfficientNet_V2_L_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    regnet_y_128gf, RegNet_Y_128GF_Weights,
    resnext101_64x4d, ResNeXt101_64X4D_Weights,
    wide_resnet101_2, Wide_ResNet101_2_Weights,
    resnet152, ResNet152_Weights
)

# 設定隨機種子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==================== 訓練與驗證函數 ====================
def get_scheduler_with_warmup(optimizer, num_epochs, warmup_epochs, num_steps_per_epoch):
    """
    建立帶有 warmup 的學習率調度器
    """
    def lr_lambda(current_step):
        # 計算當前 epoch (浮點數)
        current_epoch = current_step / num_steps_per_epoch
        
        # Warmup 階段：線性增長
        if current_epoch < warmup_epochs:
            return current_epoch / warmup_epochs
        
        # Warmup 後：Cosine Annealing
        progress = (current_epoch - warmup_epochs) / (num_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ==================== 繪製訓練曲線 ====================
def plot_training_curves(history, output_dir):
    """繪製訓練過程曲線"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 訓練損失
    axes[0, 0].plot(history['train_loss'], label='Total Loss', linewidth=2)
    axes[0, 0].plot(history['soft_loss'], label='Soft Loss', linestyle='--')
    axes[0, 0].plot(history['hard_loss'], label='Hard Loss', linestyle='--')
    axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 訓練準確率
    axes[0, 1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[0, 1].set_title('Training Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 驗證準確率
    axes[0, 2].plot(history['val_acc'], label='Val Acc', color='orange', linewidth=2)
    axes[0, 2].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 驗證 F1
    axes[1, 0].plot(history['val_f1'], label='Val F1', color='green', linewidth=2)
    axes[1, 0].set_title('Validation F1-Score', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 損失權重比較
    total_loss = np.array(history['train_loss'])
    soft_ratio = np.array(history['soft_loss']) / total_loss
    hard_ratio = np.array(history['hard_loss']) / total_loss
    
    axes[1, 1].plot(soft_ratio, label='Soft Loss Ratio', linewidth=2)
    axes[1, 1].plot(hard_ratio, label='Hard Loss Ratio', linewidth=2)
    axes[1, 1].set_title('Loss Component Ratios', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Ratio')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Accuracy 比較
    axes[1, 2].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1, 2].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1, 2].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    print(f"訓練曲線已保存: training_curves.png")
    plt.close()

def get_model_and_weights(model_name):
    """根據模型名稱返回模型和權重"""
    models_config = {
        'RegNet_Y_128GF': (regnet_y_128gf, RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1),
        'EfficientNet_V2_L': (efficientnet_v2_l, EfficientNet_V2_L_Weights.IMAGENET1K_V1),
        'ConvNeXt_Large': (convnext_large, ConvNeXt_Large_Weights.IMAGENET1K_V1),
        'ResNeXt101_64x4d': (resnext101_64x4d, ResNeXt101_64X4D_Weights.IMAGENET1K_V1),
        'Wide_ResNet101_2': (wide_resnet101_2, Wide_ResNet101_2_Weights.IMAGENET1K_V2),
        'ResNet152': (resnet152, ResNet152_Weights.IMAGENET1K_V2),
        'MobileNet_V3_Large': (mobilenet_v3_large, MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    }
    return models_config[model_name]

def modify_classifier(model, model_name, num_classes):
    """修改模型的分類層"""
    if 'ConvNeXt' in model_name:
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    elif 'EfficientNet' in model_name:
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif 'MobileNet' in model_name:
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif 'RegNet' in model_name:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif 'ResNeXt' in model_name:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    
    return model

def create_layer_wise_lr_groups(model, base_lr, llrd_rate=0.9):
    """
    創建 Layer-wise Learning Rate Decay 的參數組
    
    Args:
        model: timm 模型
        base_lr: 基礎學習率
        llrd_rate: 層級衰減率
    
    Returns:
        parameter_groups: 參數組列表
    """
    parameter_groups = []
    
    # 獲取模型的所有層
    # ConvNeXt V2 的結構: stem -> stages (4個) -> head
    num_layers = len(model.stages)
    
    # 為每一層設置不同的學習率
    # 越深的層使用越高的學習率
    layer_lrs = {}
    
    # Stem (最底層)
    layer_lrs['stem'] = base_lr * (llrd_rate ** num_layers)
    
    # Stages (中間層)
    for i in range(num_layers):
        layer_lrs[f'stages.{i}'] = base_lr * (llrd_rate ** (num_layers - i - 1))
    
    # Head (頂層 - 分類器)
    layer_lrs['head'] = base_lr
    
    # 根據層級分配參數
    params_dict = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # 確定參數屬於哪一層
        lr = base_lr  # 默認學習率
        for layer_name, layer_lr in layer_lrs.items():
            if name.startswith(layer_name):
                lr = layer_lr
                break
        
        if lr not in params_dict:
            params_dict[lr] = []
        params_dict[lr].append(param)
    
    # 創建參數組
    for lr, params in params_dict.items():
        parameter_groups.append({'params': params, 'lr': lr})
    
    print(f"創建了 {len(parameter_groups)} 個學習率組，範圍從 {min(params_dict.keys()):.2e} 到 {max(params_dict.keys()):.2e}")
    
    return parameter_groups