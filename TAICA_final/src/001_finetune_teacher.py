"""
Author: Chi-An Chen
Date: 2025-12-02
Version: 3.0
Description: Training script for comparing ConvNeXt V2 Large and Huge models
             on small agricultural dataset to select best teacher model.
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import timm

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score

from utils.dataloader import CitrusDataset, get_transforms
from utils.util import create_layer_wise_lr_groups

# ==================== 超參數配置 ====================
CONFIG = {
    'root_dir': r"C:\Users\User\Desktop\TAICA_final\TAICA期末資料集",
    'train_txt': "./TAICA期末資料集/train.txt",
    'val_txt': "./TAICA期末資料集/valid.txt",
    'test_txt': "./TAICA期末資料集/test.txt",
    
    # 模型配置
    'models': [
        'convnextv2_huge',
        'convnextv2_large'
    ],
    
    # 訓練配置
    'num_epochs': 50,
    'batch_size':16,
    'base_lr': 5e-5,
    'weight_decay': 0.05,
    'warmup_epochs': 5,
    
    # LLRD 配置
    'llrd_rate': 0.85,
    
    # 其他配置
    'image_size': 224,
    'num_workers': 4,
    'seed': 666
}


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """訓練一個 epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """驗證模型"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return epoch_loss, epoch_acc, macro_f1, all_preds, all_labels


def evaluate_model(model, dataloader, device, class_names):
    """詳細評估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 計算各項指標
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    # 計算每個類別的指標
    precision_per_class, recall_per_class, f1_per_class, support = \
        precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
    
    results = {
        'accuracy': float(accuracy),
        'precision_weighted': float(precision),
        'recall_weighted': float(recall),
        'f1_weighted': float(f1),
        'f1_macro': float(macro_f1),
        'per_class_metrics': {}
    }
    
    for i, class_name in enumerate(class_names):
        results['per_class_metrics'][class_name] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1_score': float(f1_per_class[i]),
            'support': int(support[i])
        }
    
    return results, all_preds, all_labels


def train_model(model_name, config, train_loader, val_loader, class_names, device):
    """訓練單個模型"""
    print(f"\n{'='*80}")
    print(f"開始訓練模型: {model_name}")
    print(f"{'='*80}\n")
    
    # 創建模型
    model = timm.create_model(model_name, pretrained=True, num_classes=len(class_names))
    model = model.to(device)
    
    # 計算模型參數量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型參數量: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # 創建 Layer-wise LR 參數組
    param_groups = create_layer_wise_lr_groups(
        model, 
        base_lr=config['base_lr'],
        llrd_rate=config['llrd_rate']
    )
    
    # 優化器
    optimizer = optim.AdamW(
        param_groups,
        weight_decay=config['weight_decay']
    )
    
    # 學習率調度器 - Cosine Annealing with Warmup
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'] - config['warmup_epochs'],
        eta_min=config['base_lr'] * 0.01
    )
    
    # Warmup scheduler
    warmup_scheduler = None
    if config['warmup_epochs'] > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=config['warmup_epochs']
        )
    
    # 損失函數
    criterion = nn.CrossEntropyLoss()
    
    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    # 訓練循環
    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 50)
        
        # 訓練
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # 驗證
        val_loss, val_acc, val_f1, _, _ = validate(
            model, val_loader, criterion, device
        )
        
        # 記錄
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Macro F1: {val_f1:.4f}")
        
        # 調整學習率
        if warmup_scheduler is not None and epoch < config['warmup_epochs']:
            warmup_scheduler.step()
            print(f"Warmup - LR: {optimizer.param_groups[0]['lr']:.6f}")
        else:
            scheduler.step()
            print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型（基於 Macro F1）
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch + 1
            
            model_filename = model_name.replace('.', '_').replace('/', '_')
            best_model_path = f"./checkpoints_finetune/best_{model_filename}.pth"
            os.makedirs("./checkpoints_finetune", exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_name': model_name,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'class_names': class_names,
                'config': config
            }, best_model_path)
            print(f"✓ 保存最佳模型 (Epoch {epoch+1}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f})")
    
    print(f"\n模型訓練完成!")
    print(f"最佳 Epoch: {best_epoch}")
    print(f"最佳驗證準確率: {best_val_acc:.4f}")
    print(f"最佳驗證 Macro F1: {best_val_f1:.4f}")
    
    # 清理記憶體
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return best_model_path, history, best_val_acc, best_val_f1


def main():
    """主函數"""
    # 設置隨機種子
    torch.manual_seed(CONFIG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CONFIG['seed'])
    
    # 設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # 準備數據
    print("\n準備資料集...")
    train_transform, val_transform = get_transforms(CONFIG['image_size'])
    
    train_dataset = CitrusDataset(
        CONFIG['train_txt'], 
        CONFIG['root_dir'], 
        transform=train_transform
    )
    val_dataset = CitrusDataset(
        CONFIG['val_txt'], 
        CONFIG['root_dir'], 
        transform=val_transform
    )
    test_dataset = CitrusDataset(
        CONFIG['test_txt'], 
        CONFIG['root_dir'], 
        transform=val_transform
    )
    
    print(f"類別數量: {len(train_dataset.classes)}")
    print(f"訓練樣本數: {len(train_dataset)}")
    print(f"驗證樣本數: {len(val_dataset)}")
    print(f"測試樣本數: {len(test_dataset)}")
    print(f"\n類別列表: {train_dataset.classes}")
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    # 訓練所有模型
    all_results = {}
    model_paths = {}
    
    for model_name in CONFIG['models']:
        try:
            best_path, history, best_acc, best_f1 = train_model(
                model_name,
                CONFIG,
                train_loader,
                val_loader,
                train_dataset.classes,
                device
            )
            
            model_paths[model_name] = best_path
            all_results[model_name] = {
                'best_val_acc': best_acc,
                'best_val_f1': best_f1,
                'history': history
            }
            
        except Exception as e:
            print(f"\n訓練 {model_name} 時發生錯誤: {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    
    # 在測試集上評估所有模型
    print(f"\n{'='*80}")
    print("在測試集上評估所有模型")
    print(f"{'='*80}\n")
    
    test_results = {}
    
    for model_name, model_path in model_paths.items():
        print(f"\n評估 {model_name}...")
        
        # 載入模型
        checkpoint = torch.load(model_path)
        model = timm.create_model(
            model_name, 
            pretrained=False, 
            num_classes=len(train_dataset.classes)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # 評估
        results, preds, labels = evaluate_model(
            model, 
            test_loader, 
            device, 
            train_dataset.classes
        )
        test_results[model_name] = results
        
        print(f"\n{model_name} 測試結果:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Precision (Weighted): {results['precision_weighted']:.4f}")
        print(f"  Recall (Weighted): {results['recall_weighted']:.4f}")
        print(f"  F1-Score (Weighted): {results['f1_weighted']:.4f}")
        print(f"  F1-Score (Macro): {results['f1_macro']:.4f}")
        
        # 清理
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 決定最佳模型
    print(f"\n{'='*80}")
    print("模型比較總結")
    print(f"{'='*80}\n")
    
    print(f"{'模型':<50} {'驗證Acc':<12} {'驗證F1':<12} {'測試Acc':<12} {'測試F1':<12}")
    print("-" * 100)
    
    best_model_name = None
    best_test_f1 = 0.0
    
    for model_name in CONFIG['models']:
        if model_name in all_results and model_name in test_results:
            val_acc = all_results[model_name]['best_val_acc']
            val_f1 = all_results[model_name]['best_val_f1']
            test_acc = test_results[model_name]['accuracy']
            test_f1 = test_results[model_name]['f1_macro']
            
            print(f"{model_name:<50} {val_acc:<12.4f} {val_f1:<12.4f} {test_acc:<12.4f} {test_f1:<12.4f}")
            
            if test_f1 > best_test_f1:
                best_test_f1 = test_f1
                best_model_name = model_name
    
    # 保存最佳模型
    if best_model_name:
        print(f"\n{'='*80}")
        print(f"最佳Teacher模型: {best_model_name}")
        print(f"測試集 Macro F1: {best_test_f1:.4f}")
        print(f"{'='*80}")
        
        # 複製最佳模型為 teacher model
        import shutil
        src_path = model_paths[best_model_name]
        dst_path = f"./checkpoints_finetune/best_teacher_model_{best_model_name.split('.')[0]}.pth"
        shutil.copy(src_path, dst_path)
        print(f"\n已保存最佳Teacher模型至: {dst_path}")
    
    # 保存詳細結果
    final_results = {
        'config': CONFIG,
        'validation_results': all_results,
        'test_results': test_results,
        'best_model': best_model_name,
        'best_test_f1': best_test_f1
    }
    
    with open('./results/teacher_model_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
    
    print(f"\n詳細結果已保存至: ./results/teacher_model_comparison.json")
    print("\n訓練完成!")


if __name__ == "__main__":
    main()