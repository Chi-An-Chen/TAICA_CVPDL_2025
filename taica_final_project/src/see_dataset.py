"""
Author: Chi-An Chen
Date: 2025-12-06
Description: Plot and save class distribution for citrus dataset
"""
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os

def plot_class_distribution(txt_file, root_dir, save_path='class_distribution.png', figsize=(15, 8)):
    """
    Plot and save class distribution
    
    Args:
        txt_file: Path to dataset txt file
        root_dir: Dataset root directory
        save_path: Path to save the figure
        figsize: Figure size
    """
    # Read txt file
    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Count classes
    class_names = []
    for line in lines:
        img_path = line.strip()
        parts = img_path.split('/')
        if len(parts) >= 3:
            category = parts[1]  # disease/health/pest
            subclass = parts[2]   # specific class
            class_name = f"{category}_{subclass}"
            class_names.append(class_name)
    
    # Calculate count for each class
    class_counts = Counter(class_names)
    classes = sorted(class_counts.keys())
    counts = [class_counts[cls] for cls in classes]
    
    # Set font
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Plot bar chart
    fig, ax = plt.subplots(figsize=figsize)
    x_pos = np.arange(len(classes))
    bars = ax.bar(x_pos, counts, color='steelblue', alpha=0.8, edgecolor='black')
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Set labels and title
    ax.set_xlabel('Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=14, fontweight='bold')
    ax.set_title('Citrus Dataset Class Distribution', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=10)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add statistics info
    total_samples = sum(counts)
    info_text = f'Total Samples: {total_samples}\nNumber of Classes: {len(classes)}\nAverage per Class: {total_samples/len(classes):.1f}'
    ax.text(0.98, 0.97, info_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Class distribution plot saved to: {save_path}")
    
    # Show figure
    plt.show()
    
    # Return statistics
    return {
        'classes': classes,
        'counts': counts,
        'total_samples': total_samples,
        'class_distribution': dict(zip(classes, counts))
    }


def plot_multi_split_distribution(train_txt, val_txt, test_txt, root_dir, 
                                   save_path='split_distribution.png', figsize=(18, 8)):
    """
    Plot class distribution comparison for train/val/test splits
    
    Args:
        train_txt: Path to train txt file
        val_txt: Path to validation txt file
        test_txt: Path to test txt file
        root_dir: Dataset root directory
        save_path: Path to save the figure
        figsize: Figure size
    """
    def get_class_counts(txt_file):
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        class_names = []
        for line in lines:
            img_path = line.strip()
            parts = img_path.split('/')
            if len(parts) >= 3:
                category = parts[1]
                subclass = parts[2]
                class_name = f"{category}_{subclass}"
                class_names.append(class_name)
        
        return Counter(class_names)
    
    # Get class counts for each split
    train_counts = get_class_counts(train_txt)
    val_counts = get_class_counts(val_txt)
    test_counts = get_class_counts(test_txt)
    
    # Get all classes
    all_classes = sorted(set(train_counts.keys()) | set(val_counts.keys()) | set(test_counts.keys()))
    
    # Prepare data
    train_data = [train_counts.get(cls, 0) for cls in all_classes]
    val_data = [val_counts.get(cls, 0) for cls in all_classes]
    test_data = [test_counts.get(cls, 0) for cls in all_classes]
    
    # Set font
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Plot grouped bar chart
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(all_classes))
    width = 0.25
    
    bars1 = ax.bar(x - width, train_data, width, label='Train', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x, val_data, width, label='Validation', color='#A23B72', alpha=0.8)
    bars3 = ax.bar(x + width, test_data, width, label='Test', color='#F18F01', alpha=0.8)
    
    # Set labels
    ax.set_xlabel('Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=14, fontweight='bold')
    ax.set_title('Train/Validation/Test Class Distribution Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(all_classes, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add statistics info
    info_text = (f'Train: {sum(train_data)} samples\n'
                f'Validation: {sum(val_data)} samples\n'
                f'Test: {sum(test_data)} samples\n'
                f'Total: {sum(train_data) + sum(val_data) + sum(test_data)} samples')
    ax.text(0.98, 0.97, info_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Split distribution comparison plot saved to: {save_path}")
    plt.show()

# 使用範例
if __name__ == "__main__":
    # 單一資料集分布圖
    stats = plot_class_distribution(
        txt_file='./TAICA期末資料集/test.txt',
        root_dir='./TAICA期末資料集/citrus',
        save_path='train_distribution.png'
    )
    
    # 多分割對比圖
    plot_multi_split_distribution(
        train_txt='./TAICA期末資料集/train.txt',
        val_txt='./TAICA期末資料集/valid.txt',
        test_txt='./TAICA期末資料集/test.txt',
        root_dir='./TAICA期末資料集/citrus',
        save_path='split_distribution.png'
    )