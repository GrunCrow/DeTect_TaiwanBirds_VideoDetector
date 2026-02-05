"""
Train YOLO models on all train-val split ratios automatically.
This script iterates through all generated split ratios and trains a separate model for each.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import wandb
from PRIVATE import WANDB_API_KEY
import os
import torch
import yaml
import shutil
from datetime import datetime

# ============================== CONFIGURATION ==============================
# Define training configuration here - modify these constants to change defaults

# Model configuration
MODEL_WEIGHTS = 'yolo26n.pt'           # Pretrained model: yolo26n, yolo26s, yolo26m, yolo26l, yolo26x
DEVICE = 0                              # GPU device (0, 1, 2...) or 'cpu'

# Training hyperparameters
EPOCHS = 500                            # Number of training epochs
BATCH_SIZE = 32                         # Batch size for training
PATIENCE = 50                           # Early stopping patience
SINGLE_CLS = True                       # Treat dataset as single-class
LEARNING_RATE = None                    # Initial learning rate
FINAL_LR_RATIO = None                   # Final LR as ratio of initial

# Dataset configuration
SPLITS_DIR = '../dataset/csvs/split_ratios'  # Directory containing train/val/test splits
TEST_FILE = SPLITS_DIR + '/test/test.txt'         # Test file path relative to dataset root

# Logging and output
WANDB_ENABLED = False                   # Enable Weights & Biases logging
PROJECT_NAME = 'DeTect-BMMS'           # YOLO project name for organizing results

# ============================================================================

def get_available_splits(splits_dir):
    """
    Scan the split_ratios directory and return all available train/val ratio pairs.
    
    Returns:
        List of tuples: [(ratio_name, train_file, val_file), ...]
        Example: [('95-5', 'train_95-5.txt', 'val_95-5.txt'), ...]
    """
    train_dir = splits_dir / 'train'
    val_dir = splits_dir / 'val'
    
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Split directories not found: {train_dir} or {val_dir}")
    
    # Get all train files
    train_files = sorted([f for f in train_dir.glob('train_*.txt')])
    
    splits = []
    for train_file in train_files:
        # Extract ratio from filename (e.g., train_95-5.txt -> 95-5)
        ratio = train_file.stem.replace('train_', '')
        
        # Check if corresponding val file exists
        val_file = val_dir / f'val_{ratio}.txt'
        if val_file.exists():
            splits.append((ratio, train_file, val_file))
        else:
            print(f"⚠️  Warning: No matching val file for {train_file.name}, skipping...")
    
    return splits

def create_dataset_yaml(ratio, train_file, val_file, test_file, output_dir, dataset_root):
    """
    Create a custom dataset YAML file for the given split ratio.
    
    Args:
        ratio: Ratio identifier (e.g., '95-5')
        train_file: Path to train txt file
        val_file: Path to val txt file
        test_file: Path to test txt file
        output_dir: Where to save the YAML
        dataset_root: Root directory for the dataset paths
    
    Returns:
        Path to created YAML file
    """
    yaml_path = output_dir / f'DeTect_{ratio}.yaml'
    
    # Make paths relative to dataset_root
    train_rel = train_file.relative_to(dataset_root)
    val_rel = val_file.relative_to(dataset_root)
    test_rel = test_file.relative_to(dataset_root)
    
    config = {
        'path': str(dataset_root.resolve()),
        'train': str(train_rel),
        'val': str(val_rel),
        'test': str(test_rel),
        'names': {
            0: 'bat',
            1: 'bird',
            2: 'insect',
            3: 'drone',
            4: 'plane',
            5: 'other',
            6: 'unknown'
        }
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return yaml_path

def train_model(ratio, data_yaml, model_path, device, batch_size=32, epochs=400, patience=50):
    """
    Train a YOLO model with the specified configuration.
    
    Args:
        ratio: Ratio identifier for naming (e.g., '95-5')
        data_yaml: Path to dataset YAML configuration
        model_path: Path to pretrained model weights
        device: Device to use (0 for GPU, 'cpu' for CPU)
        batch_size: Batch size for training
        epochs: Number of training epochs
        patience: Early stopping patience
    
    Returns:
        Training results object
    """
    # Create unique experiment name
    exp_name = f"splitratios_{ratio}"
    
    print(f"\n{'='*100}")
    print(f"🚀 Starting training for ratio {ratio}")
    print(f"   Experiment: {exp_name}")
    print(f"   Data config: {data_yaml}")
    print(f"   Epochs: {epochs} | Batch: {batch_size} | Device: {device}")
    print(f"{'='*100}\n")
    
    # Enable memory optimization
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Load model
    model = YOLO(model_path)
    
    # Train
    results = model.train(
        data=str(data_yaml),
        project=PROJECT_NAME,
        name=f'runs/{exp_name}',
        device=device,
        epochs=epochs,
        patience=patience,
        single_cls=SINGLE_CLS,
        batch=batch_size,
        optimizer="auto",
        imgsz=640,
        cos_lr=True,
        # lr0=LEARNING_RATE,
        # lrf=FINAL_LR_RATIO,
        # Save space by not saving intermediate weights
        save_period=-1,  # Only save last.pt and best.pt
        verbose=True,

        # ===== LOGGING ADICIONAL =====
        plots=True,                    # Genera gráficos de entrenamiento
        save_json=True,                # Guarda resultados en JSON
        save_txt=False,                # Guarda resultados en TXT (opcional)
        conf=0.1,                     # Confidence threshold para visualizaciones
        iou=0.5,                      # IoU threshold para visualizaciones
        
        # ===== AUGMENTACIÓN (para loguear efectos) =====
        # Muestra el impacto de augmentación
        # flipud=0.0,
        # fliplr=0.5,
        # mosaic=1.0,
        
        # ===== CALLBACKS PERSONALIZADOS =====
        # Logging adicional
    )
    
    print(f"\n✅ Training completed for ratio {ratio}")
    print(f"   Results saved to: DeTect-BMMS/runs/{exp_name}")
    
    # Clear cache after training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Train YOLO models on all split ratios")
    parser.add_argument('--splits-dir', type=str, default=SPLITS_DIR,
                       help=f'Directory containing train/val/test splits (default: {SPLITS_DIR})')
    parser.add_argument('--model', type=str, default=MODEL_WEIGHTS,
                       help=f'Pretrained model weights (default: {MODEL_WEIGHTS})')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help=f'Number of training epochs (default: {EPOCHS})')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE,
                       help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--device', default=DEVICE,
                       help=f'Device to use (0 for GPU 0, cpu for CPU) (default: {DEVICE})')
    parser.add_argument('--patience', type=int, default=PATIENCE,
                       help=f'Early stopping patience (default: {PATIENCE})')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                       help=f'Initial learning rate (default: {LEARNING_RATE})')
    parser.add_argument('--wandb', action='store_true', default=WANDB_ENABLED,
                       help='Enable Weights & Biases logging')
    parser.add_argument('--ratios', type=str, nargs='+',
                       help='Specific ratios to train (e.g., 95-5 80-20). If not specified, all ratios will be trained.')
    parser.add_argument('--skip-ratios', type=str, nargs='+',
                       help='Ratios to skip (e.g., 0-100 100-0)')
    args = parser.parse_args()
    
    # Convert paths to Path objects
    splits_dir = Path(args.splits_dir).resolve()
    model_path = Path(args.model)
    
    # Create output directory for temporary YAML files
    yaml_output_dir = Path('cfg/datasets/auto_generated')
    yaml_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize W&B if requested
    if args.wandb:
        wandb.login(key=WANDB_API_KEY)
    
    # Get available splits
    print(f"\n{'='*100}")
    print(f"📂 Scanning for available splits in: {splits_dir}")
    print(f"{'='*100}\n")
    
    available_splits = get_available_splits(splits_dir)
    
    if not available_splits:
        print("❌ No valid train-val split pairs found!")
        return
    
    # Filter splits based on arguments
    if args.ratios:
        available_splits = [(r, t, v) for r, t, v in available_splits if r in args.ratios]
        print(f"🎯 Training only specified ratios: {args.ratios}")
    
    if args.skip_ratios:
        available_splits = [(r, t, v) for r, t, v in available_splits if r not in args.skip_ratios]
        print(f"⏭️  Skipping ratios: {args.skip_ratios}")
    
    print(f"\n✅ Found {len(available_splits)} split ratio(s) to train:\n")
    for ratio, train_file, val_file in available_splits:
        print(f"   • Ratio {ratio:8s} | Train: {train_file.name:20s} | Val: {val_file.name}")
    
    # Get test file path (same for all ratios)
    test_file = splits_dir / 'test' / 'test.txt'
    if not test_file.exists():
        print(f"\n⚠️  Warning: Test file not found: {test_file}")
        print("   Models will be trained without test set reference")
    
    # Dataset root (parent of split_ratios)
    dataset_root = splits_dir.parent
    
    # Training loop
    print(f"\n{'='*100}")
    print(f"🎓 Starting training pipeline")
    print(f"   Model: {model_path.name}")
    print(f"   Epochs: {args.epochs} | Batch: {args.batch} | Device: {args.device} | LR: {args.lr}")
    print(f"{'='*100}\n")
    
    results_summary = []
    start_time = datetime.now()
    
    for i, (ratio, train_file, val_file) in enumerate(available_splits, 1):
        print(f"\n{'#'*100}")
        print(f"# Training {i}/{len(available_splits)}: Ratio {ratio}")
        print(f"{'#'*100}\n")
        
        try:
            # Create dataset YAML
            data_yaml = create_dataset_yaml(
                ratio, train_file, val_file, test_file, 
                yaml_output_dir, dataset_root
            )
            
            # Train model
            results = train_model(
                ratio, data_yaml, model_path,
                args.device, args.batch, args.epochs, args.patience
            )
            
            results_summary.append({
                'ratio': ratio,
                'status': 'success',
                'experiment': f"splitratios_{ratio}"
            })
            
        except Exception as e:
            print(f"\n❌ Error training ratio {ratio}: {e}")
            results_summary.append({
                'ratio': ratio,
                'status': 'failed',
                'error': str(e)
            })
            continue
    
    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*100}")
    print(f"🏁 Training pipeline completed!")
    print(f"{'='*100}\n")
    print(f"⏱️  Total duration: {duration}")
    print(f"📊 Results summary:\n")
    
    successful = sum(1 for r in results_summary if r['status'] == 'success')
    failed = sum(1 for r in results_summary if r['status'] == 'failed')
    
    print(f"   ✅ Successful: {successful}/{len(results_summary)}")
    print(f"   ❌ Failed: {failed}/{len(results_summary)}\n")
    
    if successful > 0:
        print("   Successful experiments:")
        for r in results_summary:
            if r['status'] == 'success':
                print(f"      • {r['ratio']:8s} → {r['experiment']}")
    
    if failed > 0:
        print("\n   Failed experiments:")
        for r in results_summary:
            if r['status'] == 'failed':
                print(f"      • {r['ratio']:8s} → {r.get('error', 'Unknown error')}")
    
    print(f"\n{'='*100}\n")

if __name__ == '__main__':
    main()
