"""
Author: Chi-An Chen
Date: 2025-12-05
Version: 2.1
Description: Modified to select 2 samples per class and only save when at least one model predicts correctly
             
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
from collections import defaultdict

from utils.dataloader import CitrusDataset, get_transforms


def get_target_layers(model, model_name):
    """
    Get target layers for GradCAM
    
    Args:
        model: Model
        model_name: Model name
    
    Returns:
        target_layers: List of target layers
    """
    if 'convnext' in model_name.lower():
        # ConvNeXt: Use the last block of the last stage
        target_layers = [model.stages[-1].blocks[-1].norm]
    elif 'mobilevit' in model_name.lower():
        # MobileViT: Use the last layer of the last stage
        target_layers = [model.stages[-1][-1]]
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")
    
    return target_layers


def load_model(checkpoint_path, model_name, num_classes, device):
    """Load trained model"""
    # Create model
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Successfully loaded model: {model_name}")
    print(f"  Checkpoint: {checkpoint_path}")
    if 'val_acc' in checkpoint:
        print(f"  Val Acc: {checkpoint['val_acc']:.4f}")
    if 'val_f1' in checkpoint:
        print(f"  Val F1: {checkpoint['val_f1']:.4f}")
    
    return model, checkpoint.get('class_names', None)


def generate_gradcam(model, target_layers, input_tensor, target_class=None):
    """
    Generate GradCAM using pytorch-grad-cam
    
    Args:
        model: Model
        target_layers: List of target layers
        input_tensor: Input image tensor (1, C, H, W)
        target_class: Target class, None uses predicted class
    
    Returns:
        cam: GradCAM heatmap (H, W)
        pred_class: Predicted class
        pred_prob: Prediction probability
    """
    # Create GradCAM object
    cam_algorithm = GradCAM(model=model, target_layers=target_layers)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        pred_prob = probs[0, pred_class].item()
    
    # If no target class specified, use predicted class
    if target_class is None:
        target_class = pred_class
    
    # Generate GradCAM
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam_algorithm(input_tensor=input_tensor, targets=targets)
    
    # Get CAM for first image
    cam = grayscale_cam[0, :]
    
    return cam, pred_class, pred_prob


def visualize_gradcam_comparison(original_img, input_tensor, cam1, cam2, 
                                  model1_name, model2_name,
                                  pred1, pred2, prob1, prob2, 
                                  true_label, class_name, save_path):
    """
    Visualize GradCAM comparison of two models
    
    Args:
        original_img: Original PIL Image
        input_tensor: Input tensor (for denormalization)
        cam1, cam2: GradCAM heatmaps of two models
        model1_name, model2_name: Model names
        pred1, pred2: Predicted classes
        prob1, prob2: Prediction probabilities
        true_label: True label
        class_name: True class name
        save_path: Save path
    """
    # Convert PIL Image to numpy array (0-1 range)
    img_array = np.array(original_img.resize((224, 224))) / 255.0
    
    # Generate overlay images using show_cam_on_image
    cam1_image = show_cam_on_image(img_array, cam1, use_rgb=True)
    cam2_image = show_cam_on_image(img_array, cam2, use_rgb=True)
    
    # Create comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title(f'Original Image\nTrue Label: {class_name} (Class {true_label})', 
                      fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Model 1 GradCAM
    axes[1].imshow(cam1_image)
    correct1 = 'Correct' if pred1 == true_label else 'Wrong'
    color1 = 'green' if pred1 == true_label else 'red'
    axes[1].set_title(
        f'{model1_name}\nPred: Class {pred1} (prob: {prob1:.3f}) - {correct1}',
        fontsize=12, fontweight='bold', color=color1
    )
    axes[1].axis('off')
    
    # Model 2 GradCAM
    axes[2].imshow(cam2_image)
    correct2 = 'Correct' if pred2 == true_label else 'Wrong'
    color2 = 'green' if pred2 == true_label else 'red'
    axes[2].set_title(
        f'{model2_name}\nPred: Class {pred2} (prob: {prob2:.3f}) - {correct2}',
        fontsize=12, fontweight='bold', color=color2
    )
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def organize_samples_by_class(test_dataset, image_paths):
    """
    Organize test samples by class
    
    Args:
        test_dataset: Test dataset
        image_paths: List of image paths
    
    Returns:
        class_samples: Dictionary mapping class_id to list of sample indices
    """
    class_samples = defaultdict(list)
    
    for idx in range(len(test_dataset)):
        _, label = test_dataset[idx]
        class_samples[label].append(idx)
    
    return class_samples


def main():
    """Main function"""
    # Configuration
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
        'samples_per_class': 2,  # 2 samples per class
        'output_dir': './gradcam_comparison_per_class',
        'seed': 666
    }
    
    # Set random seed
    random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CONFIG['seed'])
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Prepare data
    print("Preparing test dataset...")
    _, val_transform = get_transforms(CONFIG['image_size'])
    test_dataset = CitrusDataset(
        CONFIG['test_txt'],
        CONFIG['root_dir'],
        transform=val_transform
    )
    
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {len(test_dataset.classes)}")
    print(f"Class list: {test_dataset.classes}\n")
    
    # Load two models
    print("="*80)
    print("Loading Models")
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
    
    # Get target layers
    print("\n" + "="*80)
    print("Setting GradCAM Target Layers")
    print("="*80)
    
    target_layers1 = get_target_layers(model1, CONFIG['model1']['name'])
    target_layers2 = get_target_layers(model2, CONFIG['model2']['name'])
    
    print(f"Model 1 target layer: {target_layers1[0]}")
    print(f"Model 2 target layer: {target_layers2[0]}")
    
    # Read test image paths
    with open(CONFIG['test_txt'], 'r', encoding='utf-8') as f:
        image_paths = [line.strip().split()[0] for line in f.readlines()]
    
    # Organize samples by class
    print(f"\nOrganizing samples by class...")
    class_samples = organize_samples_by_class(test_dataset, image_paths)
    
    print(f"Samples per class:")
    for class_id, indices in sorted(class_samples.items()):
        print(f"  Class {class_id} ({test_dataset.classes[class_id]}): {len(indices)} samples")
    
    # Select samples - 2 per class
    print(f"\nSelecting {CONFIG['samples_per_class']} samples per class...")
    selected_samples = []
    
    for class_id in sorted(class_samples.keys()):
        class_name = test_dataset.classes[class_id]
        available = class_samples[class_id]
        
        # Randomly select 2 samples (or all if less than 2)
        num_to_select = min(CONFIG['samples_per_class'], len(available))
        selected = random.sample(available, num_to_select)
        
        for sample_idx in selected:
            selected_samples.append((sample_idx, class_id, class_name))
        
        print(f"  Class {class_id} ({class_name}): selected {num_to_select} samples")
    
    print(f"\nTotal selected samples: {len(selected_samples)}")
    
    # Process each sample
    print("\n" + "="*80)
    print("Generating GradCAM Comparison Images")
    print("="*80 + "\n")
    
    results = []
    saved_count = 0
    skipped_count = 0
    
    for idx, (sample_idx, true_label, class_name) in enumerate(selected_samples, 1):
        print(f"Processing sample {idx}/{len(selected_samples)} (Dataset Index: {sample_idx}, Class: {class_name})...")
        
        # Get sample
        img_tensor, label = test_dataset[sample_idx]
        img_path = os.path.join(CONFIG['root_dir'], image_paths[sample_idx])
        
        # Load original image for display
        original_img = Image.open(img_path).convert('RGB')
        
        # Prepare model input
        input_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Generate GradCAM
        try:
            cam1, pred1, prob1 = generate_gradcam(model1, target_layers1, input_tensor)
            cam2, pred2, prob2 = generate_gradcam(model2, target_layers2, input_tensor)
            
            # Check if at least one model predicts correctly
            at_least_one_correct = (pred1 == label) or (pred2 == label)
            
            if not at_least_one_correct:
                print(f"  Both models wrong - SKIPPING")
                print(f"  True: Class {label}, Model1: Class {pred1}, Model2: Class {pred2}\n")
                skipped_count += 1
                continue
            
            # Visualize and save
            save_path = os.path.join(
                CONFIG['output_dir'],
                f'gradcam_class{label:02d}_{class_name}_sample{idx:02d}.png'
            )
            
            visualize_gradcam_comparison(
                original_img, input_tensor, cam1, cam2,
                CONFIG['model1']['display_name'],
                CONFIG['model2']['display_name'],
                pred1, pred2, prob1, prob2,
                label, class_name, save_path
            )
            
            # Record result
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
            saved_count += 1
            
            print(f"  True Label: {class_name} (Class {label})")
            print(f"  {CONFIG['model1']['display_name']}: Class {pred1} (prob: {prob1:.3f}) {'CORRECT' if pred1 == label else 'WRONG'}")
            print(f"  {CONFIG['model2']['display_name']}: Class {pred2} (prob: {prob2:.3f}) {'CORRECT' if pred2 == label else 'WRONG'}")
            print(f"  Saved to: {save_path}\n")
            
        except Exception as e:
            print(f"  Processing failed: {str(e)}\n")
            continue
    
    # Statistics
    print("="*80)
    print("Statistics")
    print("="*80 + "\n")
    
    print(f"Total processed: {len(selected_samples)}")
    print(f"Saved: {saved_count}")
    print(f"Skipped (both wrong): {skipped_count}\n")
    
    if results:
        model1_correct = sum(1 for r in results if r['model1']['correct'])
        model2_correct = sum(1 for r in results if r['model2']['correct'])
        both_correct = sum(1 for r in results if r['model1']['correct'] and r['model2']['correct'])
        only_model1_correct = sum(1 for r in results if r['model1']['correct'] and not r['model2']['correct'])
        only_model2_correct = sum(1 for r in results if not r['model1']['correct'] and r['model2']['correct'])
        
        print(f"{CONFIG['model1']['display_name']}:")
        print(f"  Correct: {model1_correct}/{len(results)} ({model1_correct/len(results)*100:.1f}%)")
        print()
        print(f"{CONFIG['model2']['display_name']}:")
        print(f"  Correct: {model2_correct}/{len(results)} ({model2_correct/len(results)*100:.1f}%)")
        print()
        print(f"Both correct: {both_correct}/{len(results)} ({both_correct/len(results)*100:.1f}%)")
        print(f"Only Model 1 correct: {only_model1_correct}/{len(results)} ({only_model1_correct/len(results)*100:.1f}%)")
        print(f"Only Model 2 correct: {only_model2_correct}/{len(results)} ({only_model2_correct/len(results)*100:.1f}%)")
        
        # Save detailed results
        results_file = os.path.join(CONFIG['output_dir'], 'gradcam_comparison_results.json')
        summary = {
            'config': CONFIG,
            'statistics': {
                'total_processed': len(selected_samples),
                'total_saved': saved_count,
                'skipped': skipped_count,
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
                'only_model1_correct': only_model1_correct,
                'only_model2_correct': only_model2_correct
            },
            'samples': results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        
        print(f"\nDetailed results saved to: {results_file}")
        print(f"GradCAM comparison images saved to: {CONFIG['output_dir']}/")
    else:
        print("No successfully processed samples")
    
    print(f"\nAll processing completed!")


if __name__ == "__main__":
    main()