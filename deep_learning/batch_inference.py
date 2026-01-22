"""
Batch inference runner - Run inference on multiple models at once
"""

import argparse
import subprocess
from pathlib import Path
import sys


def find_models(base_dir="DeTect-BMMS/runs"):
    """Find all models with best.pt weights."""
    base_path = Path(base_dir)
    models = []
    
    if not base_path.exists():
        print(f"Warning: {base_dir} does not exist")
        return models
    
    for model_dir in base_path.iterdir():
        if model_dir.is_dir():
            best_weights = model_dir / "weights" / "best.pt"
            if best_weights.exists():
                models.append(model_dir)
    
    return sorted(models)


def run_inference(model_path, optimize=False, val_only=False, conf=0.25, iou=0.45, no_wandb=False):
    """Run inference on a single model."""
    cmd = ["python"]
    
    if optimize:
        cmd.append("inference_optimized.py")
    else:
        cmd.append("inference.py")
    
    cmd.extend([
        "--model-path", str(model_path),
        "--data", "cfg/datasets/DeTect.yaml",
        "--project", "DeTect-BMMS"
    ])
    
    if val_only:
        cmd.append("--val-only")
    
    if not optimize:
        cmd.extend(["--conf", str(conf)])
    
    cmd.extend(["--iou", str(iou)])
    
    if no_wandb:
        cmd.append("--no-wandb")
    
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running inference on {model_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch inference runner")
    parser.add_argument('--models-dir', type=str, default='DeTect-BMMS/runs',
                        help='Directory containing model runs')
    parser.add_argument('--models', type=str, nargs='+',
                        help='Specific model names to run (otherwise runs all)')
    parser.add_argument('--optimize', action='store_true',
                        help='Use threshold optimization')
    parser.add_argument('--val-only', action='store_true',
                        help='Run only on validation set')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold (ignored if --optimize)')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable wandb logging')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be run without executing')
    
    args = parser.parse_args()
    
    # Find all models
    all_models = find_models(args.models_dir)
    
    if not all_models:
        print(f"No models found in {args.models_dir}")
        return
    
    # Filter models if specific ones requested
    if args.models:
        models_to_run = []
        for model_name in args.models:
            matching = [m for m in all_models if model_name in m.name]
            models_to_run.extend(matching)
        
        if not models_to_run:
            print(f"No matching models found for: {args.models}")
            return
    else:
        models_to_run = all_models
    
    print(f"\n{'='*80}")
    print(f"BATCH INFERENCE RUNNER")
    print(f"{'='*80}")
    print(f"Found {len(all_models)} total models")
    print(f"Will run inference on {len(models_to_run)} models")
    print(f"Optimization: {'ENABLED' if args.optimize else 'DISABLED'}")
    if not args.optimize:
        print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    print(f"{'='*80}\n")
    
    print("Models to process:")
    for i, model in enumerate(models_to_run, 1):
        print(f"  {i}. {model.name}")
    print()
    
    if args.dry_run:
        print("DRY RUN - No inference will be executed")
        return
    
    # Confirm before running
    if len(models_to_run) > 1:
        response = input(f"Run inference on {len(models_to_run)} models? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted")
            return
    
    # Run inference on each model
    results = []
    for i, model_path in enumerate(models_to_run, 1):
        print(f"\n{'#'*80}")
        print(f"# MODEL {i}/{len(models_to_run)}: {model_path.name}")
        print(f"{'#'*80}")
        
        success = run_inference(
            model_path=model_path,
            optimize=args.optimize,
            val_only=args.val_only,
            conf=args.conf,
            iou=args.iou,
            no_wandb=args.no_wandb
        )
        
        results.append((model_path.name, success))
    
    # Print summary
    print(f"\n{'='*80}")
    print("BATCH INFERENCE SUMMARY")
    print(f"{'='*80}")
    
    successful = sum(1 for _, success in results if success)
    failed = len(results) - successful
    
    print(f"Total models: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print()
    
    if failed > 0:
        print("Failed models:")
        for model_name, success in results:
            if not success:
                print(f"  - {model_name}")
    
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
