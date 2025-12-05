"""
Author: Chi-An Chen
Date: 2025-12-02
Version: 1.0
Description: 比較 ConvNeXt V2 Atto 模型在三種訓練策略下的性能:
             1. Train from scratch (隨機初始化)
             2. Fine-tune (使用預訓練權重並訓練)
             3. Pretrained only (僅使用預訓練權重,不訓練)
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import timm

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix

from utils.dataloader import CitrusDataset, get_transforms
from utils.util import create_layer_wise_lr_groups

# ==================== 超參數配置 ====================
CONFIG = {
    'root_dir': "./TAICA期末資料集",
    'train_txt': "./TAICA期末資料集/train.txt",
    'val_txt': "./TAICA期末資料集/valid.txt",
    'test_txt': "./TAICA期末資料集/test.txt",
    
    # 模型配置
    'model_name': 'convnextv2_atto', # ConvNeXt V2 Atto 模型
    
    # 訓練配置
    'num_epochs': 50,
    'batch_size': 32,
    'base_lr': 1e-4,
    'weight_decay': 0.05,
    'warmup_epochs': 5,
    
    # LLRD 配置 (僅用於 fine-tune)
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
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 計算各項指標
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    # 計算每個類別的指標
    precision_per_class, recall_per_class, f1_per_class, support = \
        precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
    
    # 混淆矩陣
    cm = confusion_matrix(all_labels, all_preds)
    
    results = {
        'accuracy': float(accuracy),
        'precision_weighted': float(precision),
        'recall_weighted': float(recall),
        'f1_weighted': float(f1),
        'f1_macro': float(macro_f1),
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': {}
    }
    
    for i, class_name in enumerate(class_names):
        results['per_class_metrics'][class_name] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1_score': float(f1_per_class[i]),
            'support': int(support[i])
        }
    
    return results, all_preds, all_labels, all_probs


def train_model(model_name, config, train_loader, val_loader, class_names, device, 
                strategy='finetune', use_llrd=True):
    """
    訓練單個模型
    
    Args:
        strategy: 'scratch', 'finetune', 或 'pretrained_only'
        use_llrd: 是否使用 Layer-wise Learning Rate Decay (僅對 finetune 有效)
    """
    print(f"\n{'='*80}")
    print(f"訓練策略: {strategy.upper()}")
    print(f"{'='*80}\n")
    
    # 根據策略創建模型
    if strategy == 'scratch':
        # Train from scratch - 隨機初始化
        model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
        print("使用隨機初始化權重")
    elif strategy == 'finetune':
        # Fine-tune - 使用預訓練權重
        model = timm.create_model(model_name, pretrained=True, num_classes=len(class_names))
        print("使用預訓練權重進行微調")
    elif strategy == 'pretrained_only':
        # 僅使用預訓練權重,不訓練
        model = timm.create_model(model_name, pretrained=True, num_classes=len(class_names))
        print("僅使用預訓練權重(不訓練)")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    model = model.to(device)
    
    # 計算模型參數量
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"總參數量: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"可訓練參數量: {num_trainable:,} ({num_trainable/1e6:.2f}M)")
    
    # 如果是 pretrained_only,直接返回模型
    if strategy == 'pretrained_only':
        return model, None, 0.0, 0.0
    
    # 創建優化器和調度器
    if strategy == 'finetune' and use_llrd:
        # 使用 Layer-wise LR
        param_groups = create_layer_wise_lr_groups(
            model, 
            base_lr=config['base_lr'],
            llrd_rate=config['llrd_rate']
        )
        print(f"使用 Layer-wise Learning Rate Decay (rate={config['llrd_rate']})")
    else:
        # 標準優化器
        param_groups = [{'params': model.parameters(), 'lr': config['base_lr']}]
        print(f"使用統一學習率: {config['base_lr']}")
    
    optimizer = optim.AdamW(
        param_groups,
        weight_decay=config['weight_decay']
    )
    
    # 學習率調度器
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
        
        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch + 1
            
            best_model_path = f"./checkpoints_comparison_atto/best_{strategy}_model.pth"
            os.makedirs("./checkpoints_comparison_atto", exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_name': model_name,
                'strategy': strategy,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'class_names': class_names,
                'config': config
            }, best_model_path)
            print(f"✓ 保存最佳模型 (Epoch {epoch+1}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f})")
    
    print(f"\n訓練完成!")
    print(f"最佳 Epoch: {best_epoch}")
    print(f"最佳驗證準確率: {best_val_acc:.4f}")
    print(f"最佳驗證 Macro F1: {best_val_f1:.4f}")
    
    return model, history, best_val_acc, best_val_f1


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
    
    print(f"模型: {CONFIG['model_name']}")
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
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=CONFIG['num_workers'],
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    # 訓練三種策略
    strategies = ['scratch', 'finetune', 'pretrained_only']
    all_results = {}
    trained_models = {}
    
    for strategy in strategies:
        try:
            print(f"\n{'='*80}")
            print(f"開始處理策略: {strategy.upper()}")
            print(f"{'='*80}")
            
            model, history, best_acc, best_f1 = train_model(
                CONFIG['model_name'],
                CONFIG,
                train_loader,
                val_loader,
                train_dataset.classes,
                device,
                strategy=strategy
            )
            
            trained_models[strategy] = model
            
            if history is not None:
                all_results[strategy] = {
                    'best_val_acc': best_acc,
                    'best_val_f1': best_f1,
                    'history': history
                }
            else:
                all_results[strategy] = {
                    'best_val_acc': 0.0,
                    'best_val_f1': 0.0,
                    'history': None
                }
            
            # 清理GPU記憶體
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"\n處理 {strategy} 時發生錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    
    # 在測試集上評估所有模型
    print(f"\n{'='*80}")
    print("在測試集上評估所有訓練策略")
    print(f"{'='*80}\n")
    
    test_results = {}
    
    for strategy in strategies:
        if strategy not in trained_models:
            continue
            
        print(f"\n評估 {strategy.upper()} 策略...")
        
        model = trained_models[strategy]
        
        # 評估
        results, preds, labels, probs = evaluate_model(
            model, 
            test_loader, 
            device, 
            train_dataset.classes
        )
        test_results[strategy] = results
        
        print(f"\n{strategy.upper()} 測試結果:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Precision (Weighted): {results['precision_weighted']:.4f}")
        print(f"  Recall (Weighted): {results['recall_weighted']:.4f}")
        print(f"  F1-Score (Weighted): {results['f1_weighted']:.4f}")
        print(f"  F1-Score (Macro): {results['f1_macro']:.4f}")
        
        print(f"\n  每類別 F1-Score:")
        for class_name, metrics in results['per_class_metrics'].items():
            print(f"    {class_name}: {metrics['f1_score']:.4f} (support: {metrics['support']})")
        
        # 清理
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 比較結果
    print(f"\n{'='*80}")
    print("訓練策略比較總結")
    print(f"{'='*80}\n")
    
    print(f"{'策略':<20} {'驗證Acc':<12} {'驗證F1':<12} {'測試Acc':<12} {'測試F1':<12}")
    print("-" * 70)
    
    for strategy in strategies:
        if strategy in all_results and strategy in test_results:
            if all_results[strategy]['history'] is not None:
                val_acc = all_results[strategy]['best_val_acc']
                val_f1 = all_results[strategy]['best_val_f1']
            else:
                val_acc = 0.0
                val_f1 = 0.0
            
            test_acc = test_results[strategy]['accuracy']
            test_f1 = test_results[strategy]['f1_macro']
            
            print(f"{strategy:<20} {val_acc:<12.4f} {val_f1:<12.4f} {test_acc:<12.4f} {test_f1:<12.4f}")
    
    # 找出最佳策略
    best_strategy = max(test_results.items(), key=lambda x: x[1]['f1_macro'])
    print(f"\n{'='*80}")
    print(f"最佳訓練策略: {best_strategy[0].upper()}")
    print(f"測試集 Accuracy: {best_strategy[1]['accuracy']:.4f}")
    print(f"測試集 Macro F1: {best_strategy[1]['f1_macro']:.4f}")
    print(f"{'='*80}")
    
    # 保存詳細結果
    os.makedirs('./results', exist_ok=True)
    
    final_results = {
        'model_name': CONFIG['model_name'],
        'config': CONFIG,
        'validation_results': all_results,
        'test_results': test_results,
        'best_strategy': best_strategy[0],
        'best_test_acc': best_strategy[1]['accuracy'],
        'best_test_f1': best_strategy[1]['f1_macro']
    }
    
    with open('./results/atto_strategy_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
    
    print(f"\n詳細結果已保存至: ./results/atto_strategy_comparison.json")
    
    # 保存比較表格
    with open('./results/atto_strategy_comparison_table.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"ConvNeXt V2 Atto 訓練策略比較\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'策略':<20} {'驗證Acc':<12} {'驗證F1':<12} {'測試Acc':<12} {'測試F1':<12}\n")
        f.write("-"*70 + "\n")
        
        for strategy in strategies:
            if strategy in all_results and strategy in test_results:
                if all_results[strategy]['history'] is not None:
                    val_acc = all_results[strategy]['best_val_acc']
                    val_f1 = all_results[strategy]['best_val_f1']
                else:
                    val_acc = 0.0
                    val_f1 = 0.0
                
                test_acc = test_results[strategy]['accuracy']
                test_f1 = test_results[strategy]['f1_macro']
                
                f.write(f"{strategy:<20} {val_acc:<12.4f} {val_f1:<12.4f} {test_acc:<12.4f} {test_f1:<12.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write(f"最佳策略: {best_strategy[0].upper()}\n")
        f.write(f"測試集 Accuracy: {best_strategy[1]['accuracy']:.4f}\n")
        f.write(f"測試集 Macro F1: {best_strategy[1]['f1_macro']:.4f}\n")
        f.write("="*80 + "\n")
    
    print(f"比較表格已保存至: ./results/atto_strategy_comparison_table.txt")
    print("\n所有處理完成!")


if __name__ == "__main__":
    main()

"""
總參數量: 3,389,968 (3.39M)
可訓練參數量: 3,389,968 (3.39M)
"""