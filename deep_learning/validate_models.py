"""
Validate trained YOLO models on test dataset and log results to TensorBoard.
This script runs validation on all trained models and logs metrics locally.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import os
import torch
from datetime import datetime
import json

# ============================== CONFIGURATION ==============================
# Define validation configuration here

# Model and dataset paths
TRAINED_MODELS_DIR = 'DeTect-BMMS/runs'    # Directory where trained models are stored
SPLITS_DIR = '../dataset/csvs/split_ratios'
TEST_FILE = f'{SPLITS_DIR}/test/test.txt'

# Validation parameters
DEVICE = 0                              # GPU device (0, 1, 2...) or 'cpu'
BATCH_SIZE = 32                         # Batch size for validation
CONF_THRESHOLD = 0.01                   # Confidence threshold
IOU_THRESHOLD = 0.5                     # IoU threshold for NMS
IMGSZ = 640                            # Image size
SINGLE_CLS = True                       # Validate as single-class

# Logging
PROJECT_NAME = 'DeTect-BMMS'            # Project name for results organization
SAVE_JSON = True                        # Save results in JSON format
SAVE_PLOTS = True                       # Generate visualization plots

# Confidence sweep to pick best threshold (local)
ENABLE_CONF_SWEEP = True
CONF_SWEEP = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BEST_CONF_METRIC = 'map50'                 # Options: 'f1' or 'map50'

# ============================================================================

def find_best_weights(run_dir):
    """
    Find best.pt weights in a trained run directory.
    
    Args:
        run_dir: Path to the run directory
    
    Returns:
        Path to best.pt if found, else None
    """
    best_weights = run_dir / 'weights' / 'best.pt'
    if best_weights.exists():
        return best_weights
    
    last_weights = run_dir / 'weights' / 'last.pt'
    if last_weights.exists():
        return last_weights
    
    return None

def get_trained_models(models_dir):
    """
    Scan the runs directory and find all trained models.
    
    Returns:
        List of tuples: [(ratio_name, model_path), ...]
    """
    models_dir = Path(models_dir).resolve()
    
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    models = []
    
    # Look for splitratios_XX-YY directories
    for run_dir in sorted(models_dir.glob('splitratios_*')):
        if not run_dir.is_dir():
            continue
        
        # Extract ratio from directory name
        ratio = run_dir.name.replace('splitratios_', '')
        
        # Find weights
        weights = find_best_weights(run_dir)
        if weights:
            models.append((ratio, weights, run_dir))
    
    return models

def compute_f1(precision, recall):
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def pick_best_conf(metrics_by_conf, metric_name):
    best_conf = None
    best_value = -1.0
    for conf, metrics in metrics_by_conf.items():
        value = metrics.get(metric_name)
        if value is None:
            continue
        if value > best_value:
            best_value = value
            best_conf = conf
    return best_conf, best_value

def validate_model(ratio, model_path, test_yaml, device, conf, iou, batch_size, run_suffix=None):
    """
    Validate a trained YOLO model on test dataset.
    
    Args:
        ratio: Ratio identifier (e.g., '95-5')
        model_path: Path to trained model weights
        test_yaml: Path to dataset YAML with test split
        device: Device to use (0 for GPU, 'cpu' for CPU)
        output_dir: Directory to save validation results
    
    Returns:
        Validation results object
    """
    print(f"\n{'='*100}")
    print(f"🔬 Validating model for ratio {ratio}")
    print(f"   Model weights: {model_path}")
    print(f"   Test config: {test_yaml}")
    print(f"   Device: {device}")
    print(f"{'='*100}\n")
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Load model
    model = YOLO(str(model_path))
    
    # Validate on test set
    results = model.val(
        data=str(test_yaml),
        split='test',
        device=device,
        batch=batch_size,
        conf=conf,
        iou=iou,
        imgsz=IMGSZ,
        single_cls=SINGLE_CLS,
        save_json=SAVE_JSON,
        plots=SAVE_PLOTS,
        project=PROJECT_NAME,
        name=f"val/splitratios_{ratio}{run_suffix or ''}",
        verbose=True
    )
    
    print(f"\n✅ Validation completed for ratio {ratio}")
    print(f"   Results saved to: {PROJECT_NAME}/val/splitratios_{ratio}{run_suffix or ''}")
    
    # Clear cache after validation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results

def create_test_dataset_yaml(test_file, output_dir, dataset_root):
    """
    Create a dataset YAML file for test-only validation.
    
    Args:
        test_file: Path to test.txt file
        output_dir: Where to save the YAML
        dataset_root: Root directory for dataset paths
    
    Returns:
        Path to created YAML file
    """
    import yaml
    
    yaml_path = output_dir / 'DeTect_test.yaml'
    
    # Make path relative to dataset_root
    test_rel = test_file.relative_to(dataset_root)
    
    config = {
        'path': str(dataset_root.resolve()),
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

def main():
    parser = argparse.ArgumentParser(description="Validate trained YOLO models on test dataset")
    parser.add_argument('--models-dir', type=str, default=TRAINED_MODELS_DIR,
                       help=f'Directory containing trained models (default: {TRAINED_MODELS_DIR})')
    parser.add_argument('--splits-dir', type=str, default=SPLITS_DIR,
                       help=f'Directory containing splits (default: {SPLITS_DIR})')
    parser.add_argument('--device', default=DEVICE,
                       help=f'Device to use (0 for GPU 0, cpu for CPU) (default: {DEVICE})')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE,
                       help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--conf', type=float, default=CONF_THRESHOLD,
                       help=f'Confidence threshold (default: {CONF_THRESHOLD})')
    parser.add_argument('--iou', type=float, default=IOU_THRESHOLD,
                       help=f'IoU threshold for NMS (default: {IOU_THRESHOLD})')
    parser.add_argument('--ratios', type=str, nargs='+',
                       help='Specific ratios to validate (e.g., 95-5 80-20). If not specified, all will be validated.')
    parser.add_argument('--skip-ratios', type=str, nargs='+',
                       help='Ratios to skip (e.g., 0-100 100-0)')
    args = parser.parse_args()
    
    # Convert paths to Path objects
    splits_dir = Path(args.splits_dir).resolve()
    models_dir = Path(args.models_dir).resolve()
    test_file = splits_dir / 'test' / 'test.txt'
    dataset_root = splits_dir.parent
    
    # Check test file exists
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return
    
    # Create output directory for YAML files
    yaml_output_dir = Path('cfg/datasets/auto_generated')
    yaml_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test dataset YAML
    test_yaml = create_test_dataset_yaml(test_file, yaml_output_dir, dataset_root)
    
    # Find trained models
    print(f"\n{'='*100}")
    print(f"🔍 Scanning for trained models in: {models_dir}")
    print(f"{'='*100}\n")
    
    trained_models = get_trained_models(models_dir)
    
    if not trained_models:
        print("❌ No trained models found!")
        print(f"   Expected structure: {models_dir}/runs/splitratios_XX-YY/weights/best.pt")
        return
    
    # Filter models based on arguments
    if args.ratios:
        trained_models = [(r, m, d) for r, m, d in trained_models if r in args.ratios]
        print(f"🎯 Validating only specified ratios: {args.ratios}")
    
    if args.skip_ratios:
        trained_models = [(r, m, d) for r, m, d in trained_models if r not in args.skip_ratios]
        print(f"⏭️  Skipping ratios: {args.skip_ratios}")
    
    print(f"\n✅ Found {len(trained_models)} trained model(s):\n")
    for ratio, weights, run_dir in trained_models:
        print(f"   • Ratio {ratio:8s} | Weights: {weights.name:12s} | Run: {run_dir.name}")
    
    # Validation loop
    print(f"\n{'='*100}")
    print(f"🎯 Starting validation on test dataset")
    print(f"   Test file: {test_file}")
    print(f"   Batch size: {args.batch} | Device: {args.device}")
    print(f"   Confidence: {args.conf} | IoU: {args.iou}")
    if ENABLE_CONF_SWEEP:
        print(f"   Confidence sweep: {CONF_SWEEP}")
        print(f"   Best-threshold metric: {BEST_CONF_METRIC}")
    print(f"{'='*100}\n")
    
    results_summary = []
    start_time = datetime.now()
    
    for i, (ratio, weights, run_dir) in enumerate(trained_models, 1):
        print(f"\n{'#'*100}")
        print(f"# Validating {i}/{len(trained_models)}: Ratio {ratio}")
        print(f"{'#'*100}\n")
        
        try:
            metrics_by_conf = {}

            # First pass: validate at user-specified conf (e.g., 0.01)
            base_conf = args.conf
            results = validate_model(
                ratio,
                weights,
                test_yaml,
                args.device,
                conf=base_conf,
                iou=args.iou,
                batch_size=args.batch,
                run_suffix=f"_conf-{base_conf:.2f}"
            )

            precision = results.results_dict.get('metrics/precision', None)
            recall = results.results_dict.get('metrics/recall', None)
            f1 = compute_f1(precision, recall)
            metrics_by_conf[base_conf] = {
                'mAP50': results.results_dict.get('metrics/mAP50', None),
                'mAP50-95': results.results_dict.get('metrics/mAP50-95', None),
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

            # Sweep confidence thresholds to pick best
            if ENABLE_CONF_SWEEP:
                sweep_confs = list(dict.fromkeys([base_conf] + CONF_SWEEP))
                for conf in sweep_confs:
                    if conf == base_conf:
                        continue
                    sweep_results = validate_model(
                        ratio,
                        weights,
                        test_yaml,
                        args.device,
                        conf=conf,
                        iou=args.iou,
                        batch_size=args.batch,
                        run_suffix=f"_conf-{conf:.2f}"
                    )

                    precision = sweep_results.results_dict.get('metrics/precision', None)
                    recall = sweep_results.results_dict.get('metrics/recall', None)
                    f1 = compute_f1(precision, recall)
                    metrics_by_conf[conf] = {
                        'mAP50': sweep_results.results_dict.get('metrics/mAP50', None),
                        'mAP50-95': sweep_results.results_dict.get('metrics/mAP50-95', None),
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    }

                metric_key = 'f1' if BEST_CONF_METRIC == 'f1' else 'mAP50'
                best_conf, best_value = pick_best_conf(metrics_by_conf, metric_key)

                if best_conf is not None:
                    best_results = validate_model(
                        ratio,
                        weights,
                        test_yaml,
                        args.device,
                        conf=best_conf,
                        iou=args.iou,
                        batch_size=args.batch,
                        run_suffix=f"_bestconf-{best_conf:.2f}"
                    )

                    precision = best_results.results_dict.get('metrics/precision', None)
                    recall = best_results.results_dict.get('metrics/recall', None)
                    f1 = compute_f1(precision, recall)

                    results_summary.append({
                        'ratio': ratio,
                        'status': 'success',
                        'best_conf': best_conf,
                        'best_metric': metric_key,
                        'best_metric_value': best_value,
                        'mAP50': best_results.results_dict.get('metrics/mAP50', None),
                        'mAP50-95': best_results.results_dict.get('metrics/mAP50-95', None),
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    })
                else:
                    results_summary.append({
                        'ratio': ratio,
                        'status': 'success',
                        'mAP50': results.results_dict.get('metrics/mAP50', None),
                        'mAP50-95': results.results_dict.get('metrics/mAP50-95', None),
                        'precision': metrics_by_conf[base_conf]['precision'],
                        'recall': metrics_by_conf[base_conf]['recall'],
                        'f1': metrics_by_conf[base_conf]['f1']
                    })
            else:
                results_summary.append({
                    'ratio': ratio,
                    'status': 'success',
                    'mAP50': results.results_dict.get('metrics/mAP50', None),
                    'mAP50-95': results.results_dict.get('metrics/mAP50-95', None),
                    'precision': metrics_by_conf[base_conf]['precision'],
                    'recall': metrics_by_conf[base_conf]['recall'],
                    'f1': metrics_by_conf[base_conf]['f1']
                })
            
        except Exception as e:
            print(f"\n❌ Error validating ratio {ratio}: {e}")
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
    print(f"🏁 Validation pipeline completed!")
    print(f"{'='*100}\n")
    print(f"⏱️  Total duration: {duration}")
    print(f"📊 Results summary:\n")
    
    successful = sum(1 for r in results_summary if r.get('status') == 'success')
    failed = sum(1 for r in results_summary if r.get('status') == 'failed')
    
    print(f"   ✅ Successful: {successful}/{len(results_summary)}")
    print(f"   ❌ Failed: {failed}/{len(results_summary)}\n")
    
    if successful > 0:
        print("   Test Set Results:")
        print(f"   {'-'*90}")
        print(f"   {'Ratio':<10} {'BestConf':<10} {'mAP50':<10} {'mAP50-95':<12} {'Precision':<12} {'Recall':<10} {'F1':<8}")
        print(f"   {'-'*90}")
        
        for r in results_summary:
            if r.get('status') == 'success':
                best_conf = r.get('best_conf', args.conf)
                mAP50 = f"{r.get('mAP50', 0)*100:.2f}%" if r.get('mAP50') else "N/A"
                mAP50_95 = f"{r.get('mAP50-95', 0)*100:.2f}%" if r.get('mAP50-95') else "N/A"
                prec = f"{r.get('precision', 0)*100:.2f}%" if r.get('precision') else "N/A"
                rec = f"{r.get('recall', 0)*100:.2f}%" if r.get('recall') else "N/A"
                f1 = f"{r.get('f1', 0)*100:.2f}%" if r.get('f1') else "N/A"
                
                print(f"   {r['ratio']:<10} {best_conf:<10.2f} {mAP50:<10} {mAP50_95:<12} {prec:<12} {rec:<10} {f1:<8}")
        
        print(f"   {'-'*90}\n")
    
    if failed > 0:
        print("   Failed validations:")
        for r in results_summary:
            if r.get('status') == 'failed':
                print(f"      • {r['ratio']:8s} → {r.get('error', 'Unknown error')}")
    
    print(f"\n{'='*100}")
    print(f"📊 View results in TensorBoard:")
    print(f"   tensorboard --logdir={PROJECT_NAME}/val --port=6006")
    print(f"{'='*100}\n")

if __name__ == '__main__':
    main()
