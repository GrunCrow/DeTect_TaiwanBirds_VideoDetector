"""
Advanced Inference script with Threshold Optimization
Runs inference with low confidence first, finds optimal threshold, then applies it.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import wandb
from PRIVATE import WANDB_API_KEY
import yaml
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def load_best_model(training_run_path):
    """Load the best model from a training run."""
    run_path = Path(training_run_path)
    best_weights = run_path / "weights" / "best.pt"
    
    if not best_weights.exists():
        raise FileNotFoundError(f"Best model not found at: {best_weights}")
    
    print(f"Loading best model from: {best_weights}")
    model = YOLO(str(best_weights))
    return model


def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes in [x1, y1, x2, y2] format.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        float: IoU value
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def yolo_to_xyxy(yolo_box, img_width, img_height):
    """
    Convert YOLO format [x_center, y_center, width, height] (normalized) to [x1, y1, x2, y2] (absolute).
    
    Args:
        yolo_box: [x_center, y_center, width, height] in normalized coordinates (0-1)
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        [x1, y1, x2, y2] in absolute coordinates
    """
    x_center, y_center, width, height = yolo_box
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    return [x1, y1, x2, y2]


def load_ground_truth_labels(label_file_path, img_width, img_height):
    """
    Load ground truth labels from YOLO format file.
    
    Args:
        label_file_path (Path): Path to label file
        img_width: Image width
        img_height: Image height
    
    Returns:
        list: List of ground truth boxes in [x1, y1, x2, y2, class_id] format
    """
    if not label_file_path.exists():
        return []
    
    gt_boxes = []
    with open(label_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                box_xyxy = yolo_to_xyxy([x_center, y_center, width, height], img_width, img_height)
                gt_boxes.append(box_xyxy + [class_id])
    
    return gt_boxes


def build_label_lookup(split_file_path, dataset_root):
    """Build a mapping from image_id (file stem) to label file path based on split file entries."""
    lookup = {}
    split_file_path = Path(split_file_path)
    dataset_root = Path(dataset_root)

    if not split_file_path.exists():
        print(f"Warning: split file not found at {split_file_path}")
        return lookup

    with open(split_file_path, 'r') as f:
        for line in f:
            img_path_str = line.strip()
            if not img_path_str:
                continue

            img_path = Path(img_path_str)
            if not img_path.is_absolute():
                img_path = (dataset_root / img_path).resolve()

            image_id = img_path.stem

            # Replace the first 'images' segment with 'labels' if present
            parts = list(img_path.parts)
            for idx, part in enumerate(parts):
                if part == 'images':
                    parts[idx] = 'labels'
                    break
            label_path = Path(*parts).with_suffix('.txt')

            lookup[image_id] = label_path

    print(f"Built label lookup for {len(lookup)} images from split: {split_file_path}")
    return lookup


def analyze_predictions_and_find_threshold(predictions_path, output_dir, label_lookup, iou_threshold=0.5):
    """
    Analyze predictions from low-confidence run to find optimal threshold.
    
    Args:
        predictions_path (Path): Path to predictions JSON file
        output_dir (Path): Directory to save analysis plots
        labels_dir (Path): Directory containing YOLO format label files
        iou_threshold (float): IoU threshold for matching predictions to ground truth
        
    Returns:
        dict: Dictionary containing optimal thresholds and analysis results
    """
    print(f"\n{'='*80}")
    print("THRESHOLD OPTIMIZATION ANALYSIS")
    print(f"Using IoU threshold: {iou_threshold} for TP/FP matching")
    print(f"{'='*80}\n")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load predictions
    if not predictions_path.exists():
        print(f"Warning: Predictions file not found at {predictions_path}")
        return None
    
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)
    
    if len(predictions) == 0:
        print("Warning: No predictions found in file")
        return None
    
    print(f"Loaded {len(predictions)} predictions")
    
    # Group predictions by image
    predictions_by_image = defaultdict(list)
    for pred in predictions:
        image_id = pred.get('image_id', '')
        predictions_by_image[image_id].append(pred)
    
    print(f"Predictions span {len(predictions_by_image)} images")
    
    # Load ground truth for all images and match predictions
    all_predictions_data = []
    total_gt_boxes = 0
    
    print("\nMatching predictions to ground truth...")
    for image_id, image_preds in predictions_by_image.items():
        # Get image info from first prediction
        if len(image_preds) == 0:
            continue
        
        first_pred = image_preds[0]
        
        # Try to get image dimensions and path
        img_width = first_pred.get('image_width', 640)
        img_height = first_pred.get('image_height', 640)
        
        # Resolve label path using the split-derived lookup (images path -> labels path)
        label_path = label_lookup.get(image_id)
        if label_path is None:
            continue
        
        # Load ground truth
        gt_boxes = load_ground_truth_labels(label_path, img_width, img_height)
        total_gt_boxes += len(gt_boxes)
        
        # Match predictions to ground truth
        gt_matched = [False] * len(gt_boxes)
        
        for pred in image_preds:
            bbox = pred.get('bbox', [])
            if len(bbox) < 4:
                continue
            
            conf = pred.get('confidence', pred.get('score', 0))
            
            # bbox is typically [x, y, width, height] in absolute coordinates from ultralytics
            # Convert to [x1, y1, x2, y2]
            if len(bbox) == 4:
                x, y, w, h = bbox
                pred_box = [x, y, x + w, y + h]
            else:
                continue
            
            # Find best matching ground truth box
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                iou = calculate_iou(pred_box, gt_box[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Determine if this is a true positive
            is_tp = False
            if best_iou >= iou_threshold and best_gt_idx >= 0 and not gt_matched[best_gt_idx]:
                is_tp = True
                gt_matched[best_gt_idx] = True
            
            all_predictions_data.append({
                'confidence': conf,
                'is_tp': is_tp,
                'iou': best_iou
            })
    
    if len(all_predictions_data) == 0:
        print("Warning: No valid predictions found after processing")
        return None
    
    print(f"Matched {len(all_predictions_data)} predictions against {total_gt_boxes} ground truth boxes")
    
    # Extract confidence scores and labels
    all_confidences = np.array([p['confidence'] for p in all_predictions_data])
    all_labels = np.array([1 if p['is_tp'] else 0 for p in all_predictions_data])
    
    num_tp_total = all_labels.sum()
    num_fp_total = len(all_labels) - num_tp_total
    num_fn_total = total_gt_boxes - num_tp_total
    
    print(f"\nAt confidence threshold 0.01:")
    print(f"  True Positives: {num_tp_total}")
    print(f"  False Positives: {num_fp_total}")
    print(f"  False Negatives: {num_fn_total}")
    print(f"  Total GT boxes: {total_gt_boxes}")
    
    # Calculate metrics at different thresholds
    thresholds = np.linspace(0.01, 0.95, 100)
    precisions = []
    recalls = []
    f1_scores = []
    
    print("\nComputing metrics across confidence thresholds...")
    
    for threshold in thresholds:
        mask = all_confidences >= threshold
        
        if mask.sum() == 0:
            precisions.append(0)
            recalls.append(0)
            f1_scores.append(0)
            continue
        
        filtered_labels = all_labels[mask]
        true_positives = filtered_labels.sum()
        false_positives = len(filtered_labels) - true_positives
        # FN = total GT boxes - TP at this threshold
        false_negatives = total_gt_boxes - true_positives
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1_scores = np.array(f1_scores)
    
    # Find optimal thresholds
    optimal_f1_idx = np.argmax(f1_scores)
    optimal_f1_threshold = thresholds[optimal_f1_idx]
    optimal_f1_score = f1_scores[optimal_f1_idx]
    
    # Find threshold where precision and recall are balanced
    pr_diff = np.abs(precisions - recalls)
    balanced_idx = np.argmin(pr_diff)
    balanced_threshold = thresholds[balanced_idx]
    
    # Find threshold for high precision (e.g., precision >= 0.9)
    high_precision_indices = np.where(precisions >= 0.9)[0]
    high_precision_threshold = thresholds[high_precision_indices[0]] if len(high_precision_indices) > 0 else 0.5
    
    print(f"\n{'='*80}")
    print("OPTIMAL THRESHOLD ANALYSIS")
    print(f"{'='*80}")
    print(f"  At this threshold: P={precisions[optimal_f1_idx]:.4f}, R={recalls[optimal_f1_idx]:.4f}")
    print(f"\nAlternative Operating Points:")
    print(f"  Balanced P/R: {balanced_threshold:.3f} (P={precisions[balanced_idx]:.4f}, R={recalls[balanced_idx]:.4f})")
    print(f"  High Precision (P≥0.9): {high_precision_threshold:.3f}")
    print(f"High Precision Threshold: {high_precision_threshold:.3f}")
    print(f"{'='*80}\n")
    
    # Create visualization plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Precision-Recall vs Threshold
    axes[0, 0].plot(thresholds, precisions, label='Precision', linewidth=2)
    axes[0, 0].plot(thresholds, recalls, label='Recall', linewidth=2)
    axes[0, 0].axvline(optimal_f1_threshold, color='r', linestyle='--', label=f'Optimal F1 ({optimal_f1_threshold:.3f})')
    axes[0, 0].axvline(balanced_threshold, color='g', linestyle='--', label=f'Balanced ({balanced_threshold:.3f})')
    axes[0, 0].set_xlabel('Confidence Threshold')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Precision and Recall vs Confidence Threshold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: F1 Score vs Threshold
    axes[0, 1].plot(thresholds, f1_scores, linewidth=2, color='purple')
    axes[0, 1].axvline(optimal_f1_threshold, color='r', linestyle='--', label=f'Max F1 ({optimal_f1_threshold:.3f})')
    axes[0, 1].axhline(optimal_f1_score, color='r', linestyle=':', alpha=0.5)
    axes[0, 1].set_xlabel('Confidence Threshold')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title(f'F1 Score vs Confidence Threshold (Max={optimal_f1_score:.4f})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Precision-Recall Curve
    axes[1, 0].plot(recalls, precisions, linewidth=2)
    axes[1, 0].scatter([recalls[optimal_f1_idx]], [precisions[optimal_f1_idx]], 
                       c='red', s=100, zorder=5, label=f'Optimal F1 point')
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision-Recall Curve')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Confidence Distribution
    axes[1, 1].hist(all_confidences, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(optimal_f1_threshold, color='r', linestyle='--', 
                       label=f'Optimal F1 ({optimal_f1_threshold:.3f})', linewidth=2)
    axes[1, 1].axvline(balanced_threshold, color='g', linestyle='--', 
                       label=f'Balanced ({balanced_threshold:.3f})', linewidth=2)
    axes[1, 1].set_xlabel('Confidence Score')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Distribution of Prediction Confidence Scores')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = output_dir / 'threshold_optimization_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Threshold optimization plot saved to: {plot_path}")
    
    # Analyze low-confidence predictions for bias
    analyze_low_confidence_predictions(all_confidences, output_dir)
    
    # Save threshold analysis results
    results = {
        'optimal_f1_threshold': float(optimal_f1_threshold),
        'optimal_f1_score': float(optimal_f1_score),
        'optimal_precision': float(precisions[optimal_f1_idx]),
        'optimal_recall': float(recalls[optimal_f1_idx]),
        'balanced_threshold': float(balanced_threshold),
        'high_precision_threshold': float(high_precision_threshold),
        'total_predictions': len(all_confidences),
        'total_ground_truth': int(total_gt_boxes),
        'true_positives_at_low_conf': int(num_tp_total),
        'false_positives_at_low_conf': int(num_fp_total),
        'false_negatives_at_low_conf': int(num_fn_total),
        'iou_threshold_used': float(iou_threshold),
        'confidence_stats': {
            'mean': float(np.mean(all_confidences)),
            'median': float(np.median(all_confidences)),
            'std': float(np.std(all_confidences)),
            'min': float(np.min(all_confidences)),
            'max': float(np.max(all_confidences)),
            'q25': float(np.percentile(all_confidences, 25)),
            'q75': float(np.percentile(all_confidences, 75))
        },
        'threshold_analysis': {
            'thresholds': thresholds.tolist(),
            'precisions': precisions.tolist(),
            'recalls': recalls.tolist(),
            'f1_scores': f1_scores.tolist()
        }
    }
    
    results_path = output_dir / 'threshold_analysis_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Threshold analysis results saved to: {results_path}")
    
    return results


def analyze_low_confidence_predictions(confidences, output_dir):
    """Analyze predictions with low confidence scores to identify potential biases."""
    print("\nAnalyzing low-confidence predictions...")
    
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bin_counts = []
    
    for i in range(len(bins) - 1):
        count = np.sum((confidences >= bins[i]) & (confidences < bins[i+1]))
        bin_counts.append(count)
    
    # Create confidence distribution plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bin_labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(bins)-1)]
    bars = ax.bar(bin_labels, bin_counts, edgecolor='black', alpha=0.7)
    
    # Color bars by confidence level
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xlabel('Confidence Score Range')
    ax.set_ylabel('Number of Predictions')
    ax.set_title('Distribution of Predictions by Confidence Score Range')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    total = len(confidences)
    for i, (bar, count) in enumerate(zip(bars, bin_counts)):
        if count > 0:
            percentage = (count / total) * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01,
                   f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_path = output_dir / 'confidence_distribution_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confidence distribution analysis saved to: {plot_path}")
    
    # Report low-confidence statistics
    low_conf_threshold = 0.3
    low_conf_count = np.sum(confidences < low_conf_threshold)
    low_conf_percentage = (low_conf_count / len(confidences)) * 100
    
    print(f"\nLow-Confidence Predictions Analysis:")
    print(f"  Predictions with confidence < {low_conf_threshold}: {low_conf_count} ({low_conf_percentage:.2f}%)")
    print(f"  Mean confidence: {np.mean(confidences):.4f}")
    print(f"  Median confidence: {np.median(confidences):.4f}")
    
    # Save detailed statistics
    stats = {
        'low_confidence_threshold': low_conf_threshold,
        'low_confidence_count': int(low_conf_count),
        'low_confidence_percentage': float(low_conf_percentage),
        'bin_distribution': {label: int(count) for label, count in zip(bin_labels, bin_counts)},
        'statistics': {
            'mean': float(np.mean(confidences)),
            'median': float(np.median(confidences)),
            'std': float(np.std(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences))
        }
    }
    
    stats_path = output_dir / 'low_confidence_analysis.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Low-confidence analysis saved to: {stats_path}")


def run_inference(model, data_yaml, split, project, exp_name, single_cls=True, conf=0.25, iou=0.45, 
                  imgsz=640, batch=16, device=None, max_det=300, augment=False, half=False):
    """Run inference on a specific dataset split."""
    print(f"\n{'='*80}")
    print(f"Running inference on {split.upper()} set")
    print(f"  conf={conf:.3f}, iou={iou:.2f}, imgsz={imgsz}, batch={batch}")
    print(f"  single_cls={single_cls}, augment={augment}, half={half}")
    if device is not None:
        print(f"  device={device}")
    print(f"{'='*80}\n")
    
    # Prepare validation arguments
    val_kwargs = {
        'data': data_yaml,
        'split': split,
        'project': project,
        'name': f'inference/{split}/{exp_name}',
        'single_cls': single_cls,
        'save_json': True,
        'save_txt': True,
        'save_conf': True,
        'plots': True,
        'conf': conf,
        'iou': iou,
        'imgsz': imgsz,
        'batch': batch,
        'max_det': max_det,
        'augment': augment,
        'half': half,
        'verbose': True
    }
    
    # Add device if specified
    if device is not None:
        val_kwargs['device'] = device
    
    metrics = model.val(**val_kwargs)
    
    return metrics


def save_metrics(metrics, output_path):
    """Save metrics to a text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(str(metrics))
    print(f"Metrics saved to: {output_path}")


def log_to_wandb(run_name, metrics_val, metrics_test, args_dict):
    """Log all metrics to Weights & Biases."""
    run = wandb.init(
        project="DeTect-BMMS-Inference",
        name=run_name,
        config=args_dict,
        tags=["inference", "threshold-optimization"]
    )
    
    # Log validation metrics
    if metrics_val is not None:
        val_results = {
            "val/precision": float(metrics_val.box.mp) if hasattr(metrics_val.box, 'mp') else 0.0,
            "val/recall": float(metrics_val.box.mr) if hasattr(metrics_val.box, 'mr') else 0.0,
            "val/mAP50": float(metrics_val.box.map50) if hasattr(metrics_val.box, 'map50') else 0.0,
            "val/mAP50-95": float(metrics_val.box.map) if hasattr(metrics_val.box, 'map') else 0.0,
        }
        wandb.log(val_results)
        print("\nValidation metrics logged to wandb")
    
    # Log test metrics
    if metrics_test is not None:
        test_results = {
            "test/precision": float(metrics_test.box.mp) if hasattr(metrics_test.box, 'mp') else 0.0,
            "test/recall": float(metrics_test.box.mr) if hasattr(metrics_test.box, 'mr') else 0.0,
            "test/mAP50": float(metrics_test.box.map50) if hasattr(metrics_test.box, 'map50') else 0.0,
            "test/mAP50-95": float(metrics_test.box.map) if hasattr(metrics_test.box, 'map') else 0.0,
        }
        wandb.log(test_results)
        print("Test metrics logged to wandb")
    
    # Create summary table
    summary_data = []
    if metrics_val is not None:
        summary_data.append([
            "Validation",
            f"{float(metrics_val.box.mp):.4f}" if hasattr(metrics_val.box, 'mp') else "N/A",
            f"{float(metrics_val.box.mr):.4f}" if hasattr(metrics_val.box, 'mr') else "N/A",
            f"{float(metrics_val.box.map50):.4f}" if hasattr(metrics_val.box, 'map50') else "N/A",
            f"{float(metrics_val.box.map):.4f}" if hasattr(metrics_val.box, 'map') else "N/A",
        ])
    
    if metrics_test is not None:
        summary_data.append([
            "Test",
            f"{float(metrics_test.box.mp):.4f}" if hasattr(metrics_test.box, 'mp') else "N/A",
            f"{float(metrics_test.box.mr):.4f}" if hasattr(metrics_test.box, 'mr') else "N/A",
            f"{float(metrics_test.box.map50):.4f}" if hasattr(metrics_test.box, 'map50') else "N/A",
            f"{float(metrics_test.box.map):.4f}" if hasattr(metrics_test.box, 'map') else "N/A",
        ])
    
    table = wandb.Table(
        columns=["Split", "Precision", "Recall", "mAP50", "mAP50-95"],
        data=summary_data
    )
    wandb.log({"metrics_summary": table})
    
    wandb.finish()
    print(f"\nWandB run finished: {run.url}")


def main():
    parser = argparse.ArgumentParser(
        description="Inference with threshold optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core arguments
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to training run directory containing best.pt')
    parser.add_argument('--data', type=str, default='cfg/datasets/DeTect.yaml',
                        help='Path to dataset yaml file')
    parser.add_argument('--project', type=str, default='DeTect-BMMS',
                        help='Project directory for results')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name (default: auto from model path)')
    
    # Detection parameters
    parser.add_argument('--single-cls', action='store_true', default=False,
                        help='Treat as single class detection (use if model trained with single_cls=True)')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='IoU threshold for NMS (0.0-1.0)')
    parser.add_argument('--initial-conf', type=float, default=0.01,
                        help='Initial low confidence for threshold optimization')
    
    # Model inference parameters
    parser.add_argument('--imgsz', '--img-size', type=int, default=640,
                        help='Image size for inference (pixels)')
    parser.add_argument('--batch', '--batch-size', type=int, default=16,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (e.g., 0, 1, cpu, 0,1,2,3). Default: auto-detect')
    parser.add_argument('--max-det', type=int, default=300,
                        help='Maximum detections per image')
    
    # Workflow control
    parser.add_argument('--val-only', action='store_true',
                        help='Run only on validation set')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable wandb logging')
    
    args = parser.parse_args()
    
    # Extract experiment name
    if args.name is None:
        args.name = Path(args.model_path).name
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.name}_optimized_{timestamp}"
    
    print(f"\n{'='*80}")
    print(f"DeTect Threshold Optimization Inference")
    print(f"{'='*80}")
    print(f"Model path: {args.model_path}")
    print(f"Experiment name: {args.name}")
    print(f"\nThreshold Optimization:")
    print(f"  Initial confidence: {args.initial_conf}")
    print(f"\nDetection Parameters:")
    print(f"  Single class: {args.single_cls}")
    print(f"  IoU threshold: {args.iou}")
    print(f"  Max detections: {args.max_det}")
    print(f"\nInference Parameters:")
    print(f"  Image size: {args.imgsz}")
    print(f"  Batch size: {args.batch}")
    print(f"  Device: {args.device if args.device else 'auto-detect'}")
    # print(f"  Augment (TTA): {args.augment}")
    # print(f"  Half precision: {args.half}")
    print(f"{'='*80}\n")
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.login(key=WANDB_API_KEY)
    
    # Load model
    model = load_best_model(args.model_path)
    
    # PHASE 1: Low confidence validation run
    print(f"\n{'='*80}")
    print("PHASE 1: LOW CONFIDENCE VALIDATION RUN")
    print(f"{'='*80}\n")
    
    metrics_val_lowconf = run_inference(
        model=model,
        data_yaml=args.data,
        split='val',
        project=args.project,
        exp_name=f"{args.name}_lowconf",
        single_cls=args.single_cls,
        conf=args.initial_conf,
        iou=args.iou,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        max_det=args.max_det,
        augment=False,
    )

    # Build label lookup from the split file (images path -> labels path)
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)

    dataset_root = Path(data_config.get('path', '.')).resolve()
    val_split = data_config.get('val', 'val.txt')
    val_split_path = Path(val_split)
    if not val_split_path.is_absolute():
        val_split_path = (dataset_root / val_split).resolve()

    label_lookup = build_label_lookup(val_split_path, dataset_root)
    if not label_lookup:
        print("Warning: label lookup is empty; threshold analysis may be incorrect.")

    print(f"PHASE 2: THRESHOLD OPTIMIZATION ANALYSIS")
    print(f"{'='*80}\n")

    val_lowconf_dir = Path(args.project) / f'inference/val/{args.name}_lowconf'
    predictions_json = val_lowconf_dir / 'predictions.json'

    analysis_output_dir = Path(args.project) / f'inference/threshold_analysis/{args.name}'
    threshold_analysis = analyze_predictions_and_find_threshold(
        predictions_path=predictions_json,
        output_dir=analysis_output_dir,
        label_lookup=label_lookup,
        iou_threshold=args.iou
    )
    
    if threshold_analysis:
        optimal_threshold = threshold_analysis['optimal_f1_threshold']
        print(f"\n{'='*80}")
        print(f"✓ OPTIMAL THRESHOLD FOUND: {optimal_threshold:.3f}")
        print(f"{'='*80}\n")
    else:
        print("Warning: Could not determine optimal threshold, using 0.25")
        optimal_threshold = 0.25
    
    # PHASE 3: Validation with optimal threshold
    print(f"\n{'='*80}")
    print(f"PHASE 3: VALIDATION WITH OPTIMAL THRESHOLD ({optimal_threshold:.3f})")
    print(f"{'='*80}\n")
    
    metrics_val = run_inference(
        model=model,
        data_yaml=args.data,
        split='val',
        project=args.project,
        exp_name=args.name,
        single_cls=args.single_cls,
        conf=optimal_threshold,
        iou=args.iou,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        max_det=args.max_det,
        augment=False,
    )
    
    val_output_path = Path(args.project) / f'inference/val/{args.name}/metrics.txt'
    save_metrics(metrics_val, val_output_path)
    
    print("\n" + "="*80)
    print("VALIDATION RESULTS (with optimal threshold)")
    print("="*80)
    print(f"Optimal Threshold: {optimal_threshold:.3f}")
    print(f"Precision: {float(metrics_val.box.mp):.4f}" if hasattr(metrics_val.box, 'mp') else "N/A")
    print(f"Recall: {float(metrics_val.box.mr):.4f}" if hasattr(metrics_val.box, 'mr') else "N/A")
    print(f"mAP50: {float(metrics_val.box.map50):.4f}" if hasattr(metrics_val.box, 'map50') else "N/A")
    print(f"mAP50-95: {float(metrics_val.box.map):.4f}" if hasattr(metrics_val.box, 'map') else "N/A")
    print("="*80 + "\n")
    
    # PHASE 4: Test with optimal threshold
    metrics_test = None
    if not args.val_only:
        print(f"\n{'='*80}")
        print(f"PHASE 4: TEST SET EVALUATION ({optimal_threshold:.3f})")
        print(f"{'='*80}\n")
        
        metrics_test = run_inference(
            model=model,
            data_yaml=args.data,
            split='test',
            project=args.project,
            exp_name=args.name,
            single_cls=args.single_cls,
            conf=optimal_threshold,
            iou=args.iou,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            max_det=args.max_det,
            augment=False
        )
        
        test_output_path = Path(args.project) / f'inference/test/{args.name}/metrics.txt'
        save_metrics(metrics_test, test_output_path)
        
        print("\n" + "="*80)
        print("TEST RESULTS")
        print("="*80)
        print(f"Applied Threshold: {optimal_threshold:.3f}")
        print(f"Precision: {float(metrics_test.box.mp):.4f}" if hasattr(metrics_test.box, 'mp') else "N/A")
        print(f"Recall: {float(metrics_test.box.mr):.4f}" if hasattr(metrics_test.box, 'mr') else "N/A")
        print(f"mAP50: {float(metrics_test.box.map50):.4f}" if hasattr(metrics_test.box, 'map50') else "N/A")
        print(f"mAP50-95: {float(metrics_test.box.map):.4f}" if hasattr(metrics_test.box, 'map') else "N/A")
        print("="*80 + "\n")
    
    # Log to wandb
    if not args.no_wandb:
        args_dict = {
            'model_path': args.model_path,
            'data': args.data,
            'optimal_threshold': optimal_threshold,
            'initial_conf': args.initial_conf,
            'iou_threshold': args.iou,
            'single_cls': args.single_cls,
            'imgsz': args.imgsz,
            'batch': args.batch,
            'device': args.device if args.device else 'auto',
            'max_det': args.max_det,
            'augment': False,
            'timestamp': timestamp
        }
        
        if threshold_analysis:
            args_dict['threshold_analysis'] = {
                'optimal_f1_threshold': threshold_analysis['optimal_f1_threshold'],
                'optimal_f1_score': threshold_analysis['optimal_f1_score'],
                'balanced_threshold': threshold_analysis['balanced_threshold']
            }
        
        log_to_wandb(run_name, metrics_val, metrics_test, args_dict)
    
    print("\n" + "="*80)
    print("✓ INFERENCE COMPLETED SUCCESSFULLY!")
    print(f"✓ Optimal threshold: {optimal_threshold:.3f}")
    print(f"✓ Analysis saved to: {analysis_output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
