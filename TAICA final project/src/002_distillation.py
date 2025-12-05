"""
Author: Chi-An Chen
Date: 2025-12-02
Version: 2.0
Description: Knowledge Distillation (KD) - 將 ConvNeXt V2 Large Teacher
             蒸餾到 ConvNeXt V2 Atto Student
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import copy
import warnings

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from utils.dis_loss import DistillationLoss
from utils.dataloader import get_dataloaders
from utils.util import set_seed, get_scheduler_with_warmup, plot_training_curves

# 過濾警告訊息
warnings.filterwarnings('ignore')

# ==================== 配置參數 ====================
class Config:
    # 路徑設定
    root_dir = "./TAICA期末資料集"
    train_txt = "./TAICA期末資料集/train.txt"
    val_txt = "./TAICA期末資料集/valid.txt"
    test_txt = "./TAICA期末資料集/test.txt"
    teacher_path = './checkpoints_finetune/best_teacher_model_convnextv2_large.pth'
    output_dir = './checkpoints_kd'
    
    # 模型設定
    teacher_model = 'convnextv2_large'
    student_model = 'convnextv2_atto'
    
    # 訓練超參數
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-4
    weight_decay = 1e-4
    warmup_epochs = 5
    
    # 蒸餾超參數
    temperature = 4.0
    alpha = 0.8  # Soft Loss 權重 (Teacher 的影響力)
    
    # 資料增強設定
    image_size = 224
    
    # 其他設定
    num_workers = 4
    seed = 666
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

set_seed(Config.seed)

# ==================== 載入 Teacher 模型 ====================
def load_teacher_model(teacher_path, model_name, num_classes):
    """
    載入已訓練的 Teacher 模型並凍結
    """
    print(f"載入 Teacher 模型: {teacher_path}")
    
    # 創建模型架構
    teacher = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes
    )
    
    # 載入權重
    checkpoint = torch.load(teacher_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        teacher.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Teacher 驗證準確率: {checkpoint.get('val_acc', 'N/A')}")
        if 'val_f1' in checkpoint:
            print(f"✓ Teacher F1 分數: {checkpoint['val_f1']:.4f}")
    else:
        teacher.load_state_dict(checkpoint)
    
    # 凍結 Teacher 權重
    for param in teacher.parameters():
        param.requires_grad = False
    
    teacher.eval()
    
    print(f"✓ Teacher 模型已載入並凍結")
    
    return teacher

def prepare_student_model(model_name, num_classes):
    """
    準備 Student 模型
    """
    print(f"載入 Student 模型: {model_name}")
    
    # 載入預訓練模型
    student = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes
    )
    
    print(f"✓ Student 模型已載入")
    
    return student

def train_epoch(teacher, student, train_loader, criterion, optimizer, scheduler, device):
    """訓練一個 Epoch"""
    student.train()
    teacher.eval()
    
    running_loss = 0.0
    running_soft_loss = 0.0
    running_hard_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc='Training')
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # 前向傳播
        with torch.no_grad():
            teacher_logits = teacher(images)
        
        student_logits = student(images)
        
        # 計算損失
        loss, soft_loss, hard_loss = criterion(student_logits, teacher_logits, labels)
        
        # 反向傳播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # 統計
        running_loss += loss.item()
        running_soft_loss += soft_loss
        running_hard_loss += hard_loss
        
        _, preds = torch.max(student_logits, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'soft': f'{soft_loss:.4f}',
            'hard': f'{hard_loss:.4f}'
        })
    
    avg_loss = running_loss / len(train_loader)
    avg_soft_loss = running_soft_loss / len(train_loader)
    avg_hard_loss = running_hard_loss / len(train_loader)
    train_acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, avg_soft_loss, avg_hard_loss, train_acc

@torch.no_grad()
def validate(model, val_loader, device):
    """驗證模型"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    pbar = tqdm(val_loader, desc='Validating')
    
    for images, labels in pbar:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
    
    # 計算指標
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return accuracy, f1

# ==================== 主訓練流程 ====================
def train_kd():
    """知識蒸餾主訓練流程"""
    
    print("=" * 80)
    print("知識蒸餾訓練 (Knowledge Distillation)")
    print("=" * 80)
    print(f"Device: {Config.device}")
    print(f"Teacher: {Config.teacher_model}")
    print(f"Student: {Config.student_model}")
    print(f"Temperature: {Config.temperature}")
    print(f"Alpha (Soft Loss Weight): {Config.alpha}")
    print("=" * 80)
    
    # 創建輸出目錄
    os.makedirs(Config.output_dir, exist_ok=True)
    
    # 載入資料
    print("\n[1/5] 載入資料集...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(Config)
    num_classes = len(class_names)
    
    print(f"類別數量: {num_classes}")
    print(f"類別列表: {class_names}")
    print(f"訓練樣本數: {len(train_loader.dataset)}")
    print(f"驗證樣本數: {len(val_loader.dataset)}")
    print(f"測試樣本數: {len(test_loader.dataset)}")
    
    # 載入 Teacher
    print("\n[2/5] 載入 Teacher 模型...")
    teacher = load_teacher_model(
        Config.teacher_path, 
        Config.teacher_model, 
        num_classes
    )
    teacher = teacher.to(Config.device)
    
    # 準備 Student
    print("\n[3/5] 準備 Student 模型...")
    student = prepare_student_model(
        Config.student_model, 
        num_classes
    )
    student = student.to(Config.device)
    
    # 計算模型大小
    teacher_params = sum(p.numel() for p in teacher.parameters()) / 1e6
    student_params = sum(p.numel() for p in student.parameters()) / 1e6
    print(f"Teacher 參數量: {teacher_params:.2f}M")
    print(f"Student 參數量: {student_params:.2f}M")
    print(f"壓縮比: {teacher_params/student_params:.2f}x")
    
    # 定義損失函數和優化器
    print("\n[4/5] 配置訓練組件...")
    criterion = DistillationLoss(
        temperature=Config.temperature,
        alpha=Config.alpha
    )
    
    optimizer = optim.AdamW(
        student.parameters(),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay
    )
    
    # 使用帶有 warmup 的 scheduler
    num_steps_per_epoch = len(train_loader)
    scheduler = get_scheduler_with_warmup(
        optimizer,
        num_epochs=Config.num_epochs,
        warmup_epochs=Config.warmup_epochs,
        num_steps_per_epoch=num_steps_per_epoch
    )
    
    # 訓練歷史記錄
    history = {
        'train_loss': [],
        'train_acc': [],
        'soft_loss': [],
        'hard_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    best_acc = 0.0
    best_f1 = 0.0
    best_student_state = None
    best_epoch = 0
    
    # 開始訓練
    print("\n[5/5] 開始訓練...")
    print("=" * 80)
    
    for epoch in range(Config.num_epochs):
        print(f"\nEpoch [{epoch+1}/{Config.num_epochs}]")
        
        # 訓練
        train_loss, soft_loss, hard_loss, train_acc = train_epoch(
            teacher, student, train_loader, criterion, optimizer, scheduler, Config.device
        )
        
        # 驗證
        val_acc, val_f1 = validate(student, val_loader, Config.device)
        
        # 取得當前學習率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 記錄歷史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['soft_loss'].append(soft_loss)
        history['hard_loss'].append(hard_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # 顯示結果
        print(f"Train Loss: {train_loss:.4f} (Soft: {soft_loss:.4f}, Hard: {hard_loss:.4f}) | Train Acc: {train_acc:.4f}")
        print(f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | LR: {current_lr:.6f}")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_f1 = val_f1
            best_epoch = epoch + 1
            best_student_state = copy.deepcopy(student.state_dict())
            print(f"✓ Best model saved! Acc: {best_acc:.4f}, F1: {best_f1:.4f}")
    
    # 保存最佳模型
    print("\n儲存模型...")
    config_dict = {
        'root_dir': Config.root_dir,
        'teacher_model': Config.teacher_model,
        'student_model': Config.student_model,
        'batch_size': Config.batch_size,
        'num_epochs': Config.num_epochs,
        'learning_rate': Config.learning_rate,
        'temperature': Config.temperature,
        'alpha': Config.alpha
    }
    
    student.load_state_dict(best_student_state)
    
    torch.save({
        'model_state_dict': best_student_state,
        'accuracy': best_acc,
        'f1_score': best_f1,
        'class_names': class_names,
        'config': config_dict,
        'best_epoch': best_epoch
    }, os.path.join(Config.output_dir, 'best_student_kd.pth'))
    print(f"✓ 已保存模型: best_student_kd.pth (Epoch {best_epoch})")
    
    # 在測試集上評估
    print("\n在測試集上評估最佳模型...")
    student = student.to(Config.device)
    test_acc, test_f1 = validate(student, test_loader, Config.device)
    print(f"測試集 Acc: {test_acc:.4f} | F1: {test_f1:.4f}")
    
    # 繪製訓練曲線
    plot_training_curves(history, Config.output_dir)
    
    # 保存訓練歷史
    config_dict_full = {
        'root_dir': str(Config.root_dir),
        'teacher_model': str(Config.teacher_model),
        'student_model': str(Config.student_model),
        'batch_size': int(Config.batch_size),
        'num_epochs': int(Config.num_epochs),
        'learning_rate': float(Config.learning_rate),
        'weight_decay': float(Config.weight_decay),
        'warmup_epochs': int(Config.warmup_epochs),
        'temperature': float(Config.temperature),
        'alpha': float(Config.alpha),
        'image_size': int(Config.image_size),
        'seed': int(Config.seed),
        'device': str(Config.device)
    }

    results_dict = {
        'config': config_dict_full,
        'class_names': class_names,
        'best_epoch': int(best_epoch),
        'best_val_acc': float(best_acc),
        'best_val_f1': float(best_f1),
        'test_acc': float(test_acc),
        'test_f1': float(test_f1),
        'history': {
            'train_loss': [float(x) for x in history['train_loss']],
            'train_acc': [float(x) for x in history['train_acc']],
            'soft_loss': [float(x) for x in history['soft_loss']],
            'hard_loss': [float(x) for x in history['hard_loss']],
            'val_acc': [float(x) for x in history['val_acc']],
            'val_f1': [float(x) for x in history['val_f1']]
        }
    }
    
    with open(os.path.join(Config.output_dir, 'training_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=4, ensure_ascii=False)
    
    # 最終報告
    print("\n" + "=" * 80)
    print("訓練完成！")
    print("=" * 80)
    print(f"訓練模式: 知識蒸餾 (Knowledge Distillation)")
    print(f"最佳 Epoch: {best_epoch}")
    print(f"最佳驗證準確率: {best_acc:.4f}")
    print(f"最佳驗證 F1: {best_f1:.4f}")
    print(f"\n測試集表現:")
    print(f"  Acc: {test_acc:.4f}")
    print(f"  F1: {test_f1:.4f}")
    print(f"\n模型壓縮:")
    print(f"  參數壓縮比: {teacher_params/student_params:.2f}x")
    print("=" * 80)
    
    return history, results_dict


# ==================== 執行訓練 ====================
if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("知識蒸餾 (KD) - ConvNeXt V2 Large → Atto")
    print("=" * 80)
    
    # 檢查 Teacher 模型是否存在
    if not os.path.exists(Config.teacher_path):
        print(f"\n錯誤: 找不到 Teacher 模型！")
        print(f"   請確認路徑: {Config.teacher_path}")
        print(f"\n請先執行 finetune_teacher.py 訓練 Teacher 模型")
        exit(1)
    
    # 開始訓練
    try:
        history, results = train_kd()
        print("\n訓練成功完成！")
        print(f"結果保存在: {Config.output_dir}")
    except Exception as e:
        print(f"\n訓練過程中發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()