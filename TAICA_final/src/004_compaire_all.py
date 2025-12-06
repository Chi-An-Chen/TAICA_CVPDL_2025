import json
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import time
import timm

# Option 1: Use adjustText (install first: pip install adjustText)
try:
    from adjustText import adjust_text
    HAS_ADJUSTTEXT = True
except ImportError:
    HAS_ADJUSTTEXT = False
    print("Tip: Install adjustText for better label placement: pip install adjustText")

# Set up font for plots
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def count_parameters(model_path):
    """Count model parameters"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        total_params = sum(p.numel() for p in state_dict.values())
        return total_params
    except Exception as e:
        print(f"Cannot read {model_path}: {e}")
        return None

def measure_fps(model_name, device='cpu', num_runs=100, warmup=10, input_size=(1, 3, 224, 224)):
    """Measure FPS for a model on specified device"""
    try:
        # Create model
        model = timm.create_model(model_name, pretrained=False, num_classes=1000)
        model = model.to(device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_size).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)
        
        # Synchronize for GPU
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Measure time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        # Calculate FPS
        total_time = end_time - start_time
        fps = num_runs / total_time
        
        return fps
    except Exception as e:
        print(f"Error measuring FPS for {model_name} on {device}: {e}")
        return None

def get_model_fps(model_name):
    """Get FPS for both CPU and GPU"""
    print(f"Measuring FPS for {model_name}...")
    
    # Measure CPU FPS
    cpu_fps = measure_fps(model_name, device='cpu', num_runs=50, warmup=5)
    
    # Measure GPU FPS if available
    gpu_fps = None
    if torch.cuda.is_available():
        gpu_fps = measure_fps(model_name, device='cuda', num_runs=100, warmup=10)
    
    return cpu_fps, gpu_fps

def load_results(result_folder='./results'):
    """Load all JSON result files"""
    results = {}
    result_path = Path(result_folder)
    
    if not result_path.exists():
        print(f"Warning: Folder '{result_folder}' not found")
        return results
    
    json_files = list(result_path.glob('*_strategy_comparison.json'))
    print(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        print(f"Reading: {json_file.name}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                model_name = data['model_name']
                results[model_name] = data
                print(f"  - Successfully loaded model: {model_name}")
        except Exception as e:
            print(f"  - Failed to read: {e}")
    
    return results

def get_model_parameters(model_name, checkpoints_base='./'):
    """Get model parameter count"""
    checkpoint_folders = { 
        'dla46x_c': 'checkpoints_comparison_dla',
        'mnasnet_small': 'checkpoints_comparison_mnasnet',
        'mobilenetv3_small_050': 'checkpoints_comparison_mobilenetv3',
        'mobilevit_xxs': 'checkpoints_comparison_mobilevit',
        'mobilevitv2_050': 'checkpoints_comparison_mobilevit_v2',
        'convnextv2_atto': 'checkpoints_comparison_atto',
        'starnet_s1': 'checkpoints_comparison_starnet',
        'edgenext_xx_small': 'checkpoints_comparison_edgenext',
    }
    
    folder = checkpoint_folders.get(model_name)
    if not folder:
        print(f"Checkpoint folder not found for model {model_name}")
        return None
    
    model_path = Path(checkpoints_base) / folder / 'best_finetune_model.pth'
    
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return None
    
    return count_parameters(model_path)

def create_individual_plots(results, output_folder='plots'):
    """Create individual scatter plots for each strategy-metric combination"""
    Path(output_folder).mkdir(exist_ok=True)
    
    strategies = ['scratch', 'finetune', 'pretrained_only']
    strategy_names = {
        'scratch': 'Training from Scratch',
        'finetune': 'Fine-tuning',
        'pretrained_only': 'Pretrained Only'
    }
    
    metrics = ['accuracy', 'f1_macro', 'precision_weighted']
    metric_names = {
        'accuracy': 'Accuracy',
        'f1_macro': 'F1 Score (Macro)',
        'precision_weighted': 'Precision (Weighted)'
    }
    
    colors = {'scratch': '#FF6B6B', 'finetune': '#4ECDC4', 'pretrained_only': '#95E1D3'}
    markers = {'scratch': 'o', 'finetune': 's', 'pretrained_only': '^'}
    
    plot_count = 0
    
    for strategy in strategies:
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            model_names = []
            params_list = []
            metric_values = []
            
            for model_name, data in results.items():
                params = get_model_parameters(model_name)
                if params is None:
                    continue
                
                test_results = data['test_results'].get(strategy, {})
                if not test_results or test_results.get(metric) is None:
                    continue
                
                model_names.append(model_name)
                params_list.append(params / 1e6)
                metric_values.append(test_results[metric])
            
            # Plot scatter
            scatter = ax.scatter(params_list, metric_values, 
                               c=colors[strategy], marker=markers[strategy],
                               s=200, alpha=0.7, edgecolors='black', linewidth=2)
            
            # Add labels using adjustText if available
            if HAS_ADJUSTTEXT:
                texts = []
                for i, name in enumerate(model_names):
                    text = ax.text(params_list[i], metric_values[i], name,
                                 fontsize=9, fontweight='bold')
                    texts.append(text)
                
                adjust_text(texts, 
                           arrowprops=dict(arrowstyle='->', color='gray', lw=0.5, alpha=0.6),
                           expand_points=(1.5, 1.5),
                           expand_text=(1.2, 1.2),
                           force_points=(0.5, 0.5),
                           force_text=(0.5, 0.5),
                           ax=ax)
            else:
                for i, name in enumerate(model_names):
                    xytext = (8, 8) if i % 2 == 0 else (8, -12)
                    ax.annotate(name, (params_list[i], metric_values[i]),
                               xytext=xytext, textcoords='offset points',
                               fontsize=8, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[strategy], alpha=0.3),
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                                             color='gray', lw=0.5, alpha=0.6))
            
            # Set title and labels
            title = f"{strategy_names[strategy]} - {metric_names[metric]}"
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Parameters (M)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Score', fontsize=12, fontweight='bold')
            
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.set_ylim(-0.05, 1.05)
            ax.set_facecolor('#f8f9fa')
            
            # Save individual plot
            filename = f"{output_folder}/{strategy}_{metric}.png"
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved: {filename}")
            plt.close()
            
            plot_count += 1
    
    print(f"\nTotal {plot_count} individual plots saved in '{output_folder}/' folder")

def create_simplified_individual_plots(results, output_folder='plots_simplified'):
    """Create simplified individual plots with only top 3 models labeled"""
    Path(output_folder).mkdir(exist_ok=True)
    
    strategies = ['scratch', 'finetune', 'pretrained_only']
    strategy_names = {
        'scratch': 'Training from Scratch',
        'finetune': 'Fine-tuning',
        'pretrained_only': 'Pretrained Only'
    }
    
    metrics = ['accuracy', 'f1_macro', 'precision_weighted']
    metric_names = {
        'accuracy': 'Accuracy',
        'f1_macro': 'F1 Score (Macro)',
        'precision_weighted': 'Precision (Weighted)'
    }
    
    colors = {'scratch': '#FF6B6B', 'finetune': '#4ECDC4', 'pretrained_only': '#95E1D3'}
    markers = {'scratch': 'o', 'finetune': 's', 'pretrained_only': '^'}
    
    plot_count = 0
    
    for strategy in strategies:
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            model_data = []
            
            for model_name, data in results.items():
                params = get_model_parameters(model_name)
                if params is None:
                    continue
                
                test_results = data['test_results'].get(strategy, {})
                if not test_results or test_results.get(metric) is None:
                    continue
                
                model_data.append({
                    'name': model_name,
                    'params': params / 1e6,
                    'score': test_results[metric]
                })
            
            # Sort by score
            model_data.sort(key=lambda x: x['score'], reverse=True)
            
            # Plot all points
            params_list = [m['params'] for m in model_data]
            scores = [m['score'] for m in model_data]
            
            ax.scatter(params_list, scores,
                      c=colors[strategy], marker=markers[strategy],
                      s=200, alpha=0.7, edgecolors='black', linewidth=2)
            
            # Label only top 3
            rank_colors = ['gold', 'silver', '#CD7F32']
            for i, model in enumerate(model_data[:3]):
                ax.annotate(f"{i+1}. {model['name']}", 
                           (model['params'], model['score']),
                           xytext=(10, 10 - i*15), textcoords='offset points',
                           fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.5', 
                                   facecolor=rank_colors[i], 
                                   alpha=0.7, edgecolor='black', linewidth=1),
                           arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
            
            # Set title and labels
            title = f"{strategy_names[strategy]} - {metric_names[metric]} (Top 3)"
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Parameters (M)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Score', fontsize=12, fontweight='bold')
            
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.set_ylim(-0.05, 1.05)
            ax.set_facecolor('#f8f9fa')
            
            # Save individual plot
            filename = f"{output_folder}/{strategy}_{metric}_top3.png"
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved: {filename}")
            plt.close()
            
            plot_count += 1
    
    print(f"\nTotal {plot_count} simplified plots saved in '{output_folder}/' folder")

def create_detailed_tables(results):
    """Create detailed result tables"""
    strategies = ['scratch', 'finetune', 'pretrained_only']
    strategy_names = {
        'scratch': 'Training from Scratch',
        'finetune': 'Fine-tuning',
        'pretrained_only': 'Pretrained Only'
    }
    
    for strategy in strategies:
        print(f"\n{'='*100}")
        print(f"{strategy_names[strategy]} - Detailed Test Results")
        print(f"{'='*100}\n")
        
        table_data = []
        
        for model_name, data in results.items():
            params = get_model_parameters(model_name)
            test_results = data['test_results'].get(strategy, {})
            
            if not test_results:
                continue
            
            row = {
                'Model Name': model_name,
                'Parameters (M)': f"{params/1e6:.2f}" if params else 'N/A',
                'Accuracy': f"{test_results.get('accuracy', 0):.4f}",
                'F1 (Weighted)': f"{test_results.get('f1_weighted', 0):.4f}",
                'F1 (Macro)': f"{test_results.get('f1_macro', 0):.4f}",
                'Precision (Weighted)': f"{test_results.get('precision_weighted', 0):.4f}",
                'Recall (Weighted)': f"{test_results.get('recall_weighted', 0):.4f}"
            }
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        if not df.empty:
            df = df.sort_values('Accuracy', ascending=False)
            print(df.to_string(index=False))
            
            csv_filename = f'table_{strategy}.csv'
            df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            print(f"\nTable saved as '{csv_filename}'")
        else:
            print(f"No test results for {strategy_names[strategy]}")

def create_summary_comparison(results, measure_fps_flag=True):
    """Create summary comparison table with FPS measurements"""
    print(f"\n{'='*120}")
    print("Summary Comparison of All Models and Strategies")
    print(f"{'='*120}\n")
    
    # Measure FPS for each model once
    fps_cache = {}
    if measure_fps_flag:
        print("\nMeasuring FPS for all models (this may take a while)...\n")
        unique_models = set(results.keys())
        for model_name in unique_models:
            cpu_fps, gpu_fps = get_model_fps(model_name)
            fps_cache[model_name] = {'cpu': cpu_fps, 'gpu': gpu_fps}
            cpu_fps_str = f"{cpu_fps:.2f}" if cpu_fps is not None else "N/A"
            gpu_fps_str = f"{gpu_fps:.2f}" if gpu_fps is not None else "N/A"
            print(f"{model_name}: CPU FPS = {cpu_fps_str}, GPU FPS = {gpu_fps_str}")
    
    table_data = []
    
    for model_name, data in results.items():
        params = get_model_parameters(model_name)
        
        for strategy in ['scratch', 'finetune', 'pretrained_only']:
            test_results = data['test_results'].get(strategy, {})
            if not test_results:
                continue
            
            row = {
                'Model Name': model_name,
                'Training Strategy': {'scratch': 'From Scratch', 'finetune': 'Fine-tuning', 'pretrained_only': 'Pretrained Only'}[strategy],
                'Parameters (M)': f"{params/1e6:.2f}" if params else 'N/A',
                'Accuracy': f"{test_results.get('accuracy', 0):.4f}",
                'F1 (Macro)': f"{test_results.get('f1_macro', 0):.4f}",
                'Precision (Weighted)': f"{test_results.get('precision_weighted', 0):.4f}"
            }
            
            # Add FPS data
            if measure_fps_flag and model_name in fps_cache:
                cpu_fps = fps_cache[model_name]['cpu']
                gpu_fps = fps_cache[model_name]['gpu']
                row['CPU FPS'] = f"{cpu_fps:.2f}" if cpu_fps is not None else 'N/A'
                row['GPU FPS'] = f"{gpu_fps:.2f}" if gpu_fps is not None else 'N/A'
            
            table_data.append(row)
    
    if not table_data:
        print("No test results available")
        return
    
    df = pd.DataFrame(table_data)
    df = df.sort_values(['Model Name', 'Accuracy'], ascending=[True, False])
    print(df.to_string(index=False))
    
    df.to_csv('summary_comparison.csv', index=False, encoding='utf-8-sig')
    print(f"\nSummary table saved as 'summary_comparison.csv'")

def main():
    print("Starting model comparison analysis...")
    
    results = load_results('results')
    
    if not results:
        print("\nError: No result files found!")
        return
    
    print(f"\nSuccessfully loaded results for {len(results)} models\n")
    
    # Create individual plots
    print("\nGenerating individual plots for each strategy-metric combination...")
    if HAS_ADJUSTTEXT:
        print("Using adjustText for optimal label placement...")
        create_individual_plots(results)
    else:
        print("Using manual label placement (install adjustText for better results)...")
        create_individual_plots(results)
    
    # Create simplified individual plots
    print("\nGenerating simplified plots (top 3 only)...")
    create_simplified_individual_plots(results)
    
    # Create detailed tables
    print("\nGenerating detailed result tables...")
    create_detailed_tables(results)
    
    # Create summary comparison with FPS
    print("\nGenerating summary comparison with FPS measurements...")
    create_summary_comparison(results, measure_fps_flag=True)
    
    print("\n\nAnalysis complete!")
    print("\nGenerated files:")
    print("1. plots/ folder - 9 individual full-labeled plots")
    print("2. plots_simplified/ folder - 9 individual top-3 labeled plots")
    print("3. table_*.csv - Detailed results for each strategy")
    print("4. summary_comparison.csv - Complete comparison with FPS data")

if __name__ == "__main__":
    main()