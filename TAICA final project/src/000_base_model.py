"""
Author: Chi-An Chen
Date: 2025-11-27
Version: 2.1
Description:
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from utils.dataloader import CitrusDataset
from utils.util import get_model_and_weights, modify_classifier

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_preds, all_labels

def evaluate_model(model, dataloader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 計算各項指標
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # 計算每個類別的指標
    precision_per_class, recall_per_class, f1_per_class, support = \
        precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
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

def train_model(model_name, root_dir, train_txt, val_txt, num_epochs=20, batch_size=32, lr=0.001):
    print(f"\n{'='*80}")
    print(f"開始訓練模型: {model_name}")
    print(f"{'='*80}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}\n")
    
    # 獲取模型和權重
    model_fn, weights = get_model_and_weights(model_name)
    
    # 資料轉換
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.34928706, 0.4075573, 0.27877277], std=[0.20200579, 0.19030167, 0.19916368])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.34928706, 0.4075573, 0.27877277], std=[0.20200579, 0.19030167, 0.19916368])
    ])
    
    # 載入資料集
    train_dataset = CitrusDataset(train_txt, root_dir, transform=train_transform)
    val_dataset = CitrusDataset(val_txt, root_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)
    
    num_classes = len(train_dataset.classes)
    print(f"類別數量: {num_classes}")
    print(f"訓練樣本數: {len(train_dataset)}")
    print(f"驗證樣本數: {len(val_dataset)}\n")
    
    # 建立模型
    model = model_fn(weights=weights)
    model = modify_classifier(model, model_name, num_classes)
    model = model.to(device)
    
    # 損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # 訓練
    best_val_acc = 0.0
    best_model_path = f"./checkpoints/best_{model_name.lower()}.pth"
    os.makedirs("./checkpoints", exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        scheduler.step(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': train_dataset.classes,
                'class_to_idx': train_dataset.class_to_idx
            }, best_model_path)
            print(f"保存最佳模型 (Val Acc: {val_acc:.4f})")
    
    # 訓練完成後清除
    class_names = train_dataset.classes
    del model
    del train_dataset
    del val_dataset
    del train_loader
    del val_loader
    del optimizer
    del scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return best_model_path, class_names

def main():
    # 配置參數
    root_dir = "./TAICA期末資料集"
    train_txt = "./TAICA期末資料集/train.txt"
    val_txt = "./TAICA期末資料集/valid.txt"
    test_txt = "./TAICA期末資料集/test.txt"
    
    model_names = [
        'ConvNeXt_Large',
        'EfficientNet_V2_L',
        'MobileNet_V3_Large',
        'RegNet_Y_128GF',
        'ResNeXt101_64x4d',
        'Wide_ResNet101_2',
        'ResNet152'
    ]
    
    # 訓練配置
    num_epochs = 50
    batch_size = 32
    lr = 1e-4
    
    all_results = {}
    
    # 訓練每個模型
    for model_name in model_names:
        try:
            # 訓練模型
            best_model_path, class_names = train_model(
                model_name, root_dir, train_txt, val_txt, 
                num_epochs=num_epochs, batch_size=batch_size, lr=lr
            )
            
            # 在測試集上評估
            print(f"\n{'='*80}")
            print(f"在測試集上評估模型: {model_name}")
            print(f"{'='*80}\n")
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 載入最佳模型
            checkpoint = torch.load(best_model_path)
            model_fn, weights = get_model_and_weights(model_name)
            model = model_fn(weights=None)
            model = modify_classifier(model, model_name, len(class_names))
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            
            # 測試資料
            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.34928706, 0.4075573, 0.27877277], std=[0.20200579, 0.19030167, 0.19916368])
            ])
            
            test_dataset = CitrusDataset(test_txt, root_dir, transform=test_transform)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            
            # 評估
            results, preds, labels = evaluate_model(model, test_loader, device, class_names)
            all_results[model_name] = results
            
            print(f"\n測試結果:")
            print(f"Accuracy: {results['accuracy']:.4f}")
            print(f"Precision: {results['precision']:.4f}")
            print(f"Recall: {results['recall']:.4f}")
            print(f"F1-Score: {results['f1_score']:.4f}")
            
            # 清除VRAM
            print(f"\n清除 {model_name} 的VRAM...")
            del model
            del checkpoint
            del test_dataset
            del test_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print("VRAM已清除\n")
            
        except Exception as e:
            print(f"訓練 {model_name} 時發生錯誤: {str(e)}")
            # 即使出錯也要清除VRAM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            continue
    
    # 保存結果
    with open('./results/all_models_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    
    # 顯示總結
    print(f"\n{'='*80}")
    print("所有模型測試結果總結")
    print(f"{'='*80}\n")
    print(f"{'模型名稱':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 80)
    
    for model_name, results in all_results.items():
        print(f"{model_name:<25} {results['accuracy']:<12.4f} {results['precision']:<12.4f} "
              f"{results['recall']:<12.4f} {results['f1_score']:<12.4f}")
    
    print("\n詳細結果已保存至 ./results/all_models_results.json")

if __name__ == "__main__":
    main()



