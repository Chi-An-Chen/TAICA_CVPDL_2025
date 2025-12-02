"""
Author: Chi-An Chen
Date: 2025-12-02
Version: 2.1
Description:
"""
import os

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class CitrusDataset(Dataset):
    """柑橘資料集類別"""
    def __init__(self, txt_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        # 讀取txt文件
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 建立類別映射
        classes = set()
        for line in lines:
            img_path = line.strip()
            parts = img_path.split('/')
            if len(parts) >= 3:
                category = parts[1]  # disease/health/pest
                subclass = parts[2]   # 具體類別
                class_name = f"{category}_{subclass}"
                classes.add(class_name)
        
        self.classes = sorted(list(classes))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # 載入樣本
        for line in lines:
            img_path = line.strip()
            parts = img_path.split('/')
            if len(parts) >= 3:
                category = parts[1]
                subclass = parts[2]
                class_name = f"{category}_{subclass}"
                label = self.class_to_idx[class_name]
                full_path = os.path.join(root_dir, img_path)
                self.samples.append((full_path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(image_size=224):
    """
    標準增強：與 Teacher 訓練時相同
    使用您資料集的標準化參數
    """
    mean = [0.34928706, 0.4075573, 0.27877277]
    std = [0.20200579, 0.19030167, 0.19916368]
    
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return train_transform, val_transform

def get_dataloaders(config):
    """載入資料集"""
    train_transform, val_transform = get_transforms(config.image_size)
    
    train_dataset = CitrusDataset(
        config.train_txt,
        config.root_dir,
        transform=train_transform
    )
    
    val_dataset = CitrusDataset(
        config.val_txt,
        config.root_dir,
        transform=val_transform
    )
    
    test_dataset = CitrusDataset(
        config.test_txt,
        config.root_dir,
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.classes


