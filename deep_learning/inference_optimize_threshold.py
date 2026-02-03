"""
Advanced Inference script with Threshold Optimization
Runs inference with low confidence first, finds optimal threshold, then applies it.

python inference_optimize_threshold.py --model-path DeTect-BMMS/runs/ --single-cls --save-plots 50 --log-images --batch 32

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
from PIL import Image, ImageDraw


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
    """Build a mapping from image_id -> {image_path, label_path} based on split entries."""
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

            lookup[image_id] = {
                'image_path': img_path,
                'label_path': label_path
            }

    print(f"Built label lookup for {len(lookup)} images from split: {split_file_path}")
    return lookup


def analyze_predictions_and_find_threshold(predictions_path, output_dir, label_lookup, iou_threshold=0.5, save_plots=0, optimize_metric='f1'):
    """
    Analyze predictions from low-confidence run to find optimal threshold.
    
    Args:
        predictions_path (Path): Path to predictions JSON file
        output_dir (Path): Directory to save analysis plots
        label_lookup (dict): Mapping from image_id to image/label paths
        iou_threshold (float): IoU threshold for matching predictions to ground truth
        save_plots (int): Number of visualization plots to save
        optimize_metric (str): Metric to optimize ('f1', 'precision', 'recall')
        
    Returns:
        dict: Dictionary containing optimal thresholds and analysis results
    """
    print(f"\n{'='*80}")
    print("THRESHOLD OPTIMIZATION ANALYSIS")
    print(f"Using IoU threshold: {iou_threshold} for TP/FP matching")
    print(f"Optimizing for: {optimize_metric.upper()}")
    print(f"{'='*80}\n")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    conf_matrix_dir = output_dir / 'conf_matrix'
    conf_matrix_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # DEBUG: Show sample prediction structure
    if len(predictions) > 0:
        print(f"\nDEBUG: Sample prediction structure:")
        sample = predictions[0]
        print(f"  Keys: {list(sample.keys())}")
        print(f"  Sample: {sample}")
    
    # Group predictions by image
    predictions_by_image = defaultdict(list)
    for pred in predictions:
        image_id = pred.get('image_id', '')
        predictions_by_image[image_id].append(pred)
    
    print(f"Predictions span {len(predictions_by_image)} images")
    print(f"Label lookup has {len(label_lookup)} entries")
    
    # Load ground truth for all images and match predictions
    all_predictions_data = []
    total_gt_boxes = 0
    images_with_gt = 0
    images_without_gt = 0
    missing_label_files = 0
    
    print("\nMatching predictions to ground truth...")
    vis_saved = 0
    for image_id, image_preds in predictions_by_image.items():
        # Get image info from first prediction
        if len(image_preds) == 0:
            continue
        
        first_pred = image_preds[0]
        
        # Resolve paths using the split-derived lookup (images path -> labels path)
        entry = label_lookup.get(image_id)
        if entry is None:
            images_without_gt += 1
            continue
        label_path = entry['label_path']
        image_path = entry.get('image_path')

        # Try to get image dimensions from prediction metadata first, otherwise from the image file
        img_width = first_pred.get('image_width')
        img_height = first_pred.get('image_height')

        if (not img_width or not img_height) and image_path and Path(image_path).exists():
            try:
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                print(f"Warning: failed to read image size for {image_id} at {image_path}: {e}")

        # Fallback to 640 if still unknown
        img_width = img_width or 640
        img_height = img_height or 640

        if not label_path.exists():
            if missing_label_files < 5:
                print(f"Warning: label file not found for {image_id}: {label_path}")
            missing_label_files += 1
            gt_boxes = []
        else:
            gt_boxes = load_ground_truth_labels(label_path, img_width, img_height)
        if len(gt_boxes) > 0:
            images_with_gt += 1
        total_gt_boxes += len(gt_boxes)
        
        # DEBUG: Show first image with GT details
        # if images_with_gt == 1 and len(gt_boxes) > 0:
        #     print(f"\nDEBUG: First image with GT:")
        #     print(f"  Image ID: {image_id}")
        #     print(f"  Image dims: {img_width}x{img_height}")
        #     print(f"  GT boxes count: {len(gt_boxes)}")
        #     print(f"  Sample GT box (xyxy): {gt_boxes[0][:4]}")
        #     print(f"  Predictions for this image: {len(image_preds)}")
        
        # Match predictions to ground truth
        gt_matched = [False] * len(gt_boxes)
        
        image_tp_count = 0
        
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
                image_tp_count += 1
            
                # DEBUG: Show first few matches
                # if images_with_gt == 1 and len(all_predictions_data) < 3:
                #     print(f"\n  DEBUG: Prediction {len(all_predictions_data)+1}:")
                #     print(f"    Pred box (xyxy): {pred_box}")
                #     print(f"    Confidence: {conf:.4f}")
                #     print(f"    Best IoU: {best_iou:.4f}")
                #     print(f"    Best GT idx: {best_gt_idx}")
                #     print(f"    IoU threshold: {iou_threshold}")
                #     print(f"    Is TP: {is_tp}")
            
            all_predictions_data.append({
                'confidence': conf,
                'is_tp': is_tp,
                'iou': best_iou
            })
        
            # DEBUG: Summary for first image
            if images_with_gt == 1:
                print(f"\n  First image summary: {image_tp_count} TPs out of {len(image_preds)} predictions")

            # Optional visualization saving
            if save_plots > 0 and vis_saved < save_plots and image_path and Path(image_path).exists():
                try:
                    img = Image.open(image_path).convert("RGB")
                    draw = ImageDraw.Draw(img)

                    # Draw GT boxes (green)
                    for gt_box in gt_boxes:
                        draw.rectangle(gt_box[:4], outline=(0, 255, 0), width=2)

                    # Draw predictions (red) with IoU/conf labels
                    for pred in image_preds:
                        bbox = pred.get('bbox', [])
                        if len(bbox) != 4:
                            continue
                        x, y, w, h = bbox
                        pred_box = [x, y, x + w, y + h]
                        conf = pred.get('confidence', pred.get('score', 0))
                        draw.rectangle(pred_box, outline=(255, 0, 0), width=2)
                        draw.text((pred_box[0], max(0, pred_box[1]-10)), f"{conf:.2f}", fill=(255,0,0))

                    vis_dir = output_dir / 'vis'
                    vis_dir.mkdir(parents=True, exist_ok=True)
                    vis_path = vis_dir / f"{image_id}.png"
                    img.save(vis_path)
                    vis_saved += 1
                except Exception as e:
                    print(f"Warning: failed to save visualization for {image_id}: {e}")
    
    if len(all_predictions_data) == 0:
        print("Warning: No valid predictions found after processing")
        return None
    
    # print(f"\nDEBUG: Matching summary:")
    # print(f"  Images processed: {len(predictions_by_image)}")
    # print(f"  Images with GT found: {images_with_gt}")
    # print(f"  Images without GT: {images_without_gt}")
    # print(f"  Missing label files: {missing_label_files}")
    # print(f"  Total predictions: {len(all_predictions_data)}")
    # print(f"  Total GT boxes: {total_gt_boxes}")
    
    if total_gt_boxes == 0:
        print("\nERROR: No ground truth boxes found!")
        print("Possible issues:")
        print("  1. Label files not found at expected paths")
        print("  2. Label files are empty")
        print("  3. image_id in predictions doesn't match label lookup keys")
        print("\nSample image_ids from predictions:", list(predictions_by_image.keys())[:5])
        print("Sample image_ids from label_lookup:", list(label_lookup.keys())[:5])
        return None
    
    print(f"\nMatched {len(all_predictions_data)} predictions against {total_gt_boxes} ground truth boxes")
    
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
    
    if num_tp_total == 0:
        print("\nWARNING: Zero true positives at threshold 0.01!")
        print(f"  IoU threshold used: {iou_threshold}")
        print(f"  Sample IoUs from predictions: {[p['iou'] for p in all_predictions_data[:10]]}")
        print("\nThis suggests:")
        print(f"  1. IoU threshold ({iou_threshold}) may be too high")
        print("  2. Bounding box coordinate format mismatch")
        print("  3. Ground truth labels may be incorrect")
    
    # Calculate metrics at different thresholds
    thresholds = np.linspace(0.01, 0.95, 100)
    precisions = []
    recalls = []
    f1_scores = []
    accuracies = []
    tprs = []
    fprs = []
    
    print("\nComputing metrics across confidence thresholds...")
    
    for threshold in thresholds:
        mask = all_confidences >= threshold
        
        if mask.sum() == 0:
            precisions.append(0)
            recalls.append(0)
            f1_scores.append(0)
            accuracies.append(0)
            tprs.append(0)
            fprs.append(0)
            continue
        
        filtered_labels = all_labels[mask]

        tp = int(filtered_labels.sum())
        fp = int(len(filtered_labels) - tp)
        fn = int((all_labels == 1).sum() - tp)
        tn = int((all_labels == 0).sum() - fp)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        tpr = recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        accuracies.append(accuracy)
        tprs.append(tpr)
        fprs.append(fpr)
    
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1_scores = np.array(f1_scores)
    accuracies = np.array(accuracies)
    tprs = np.array(tprs)
    fprs = np.array(fprs)

    # Area under curves for reference (sorted by increasing recall/FPR)
    recall_order = np.argsort(recalls)
    auc_pr = float(np.trapezoid(precisions[recall_order], recalls[recall_order]))
    fpr_order = np.argsort(fprs)
    auc_roc = float(np.trapezoid(tprs[fpr_order], fprs[fpr_order]))
    
    # Find optimal threshold based on selected metric
    metric_lookup = {
        'f1': (f1_scores, 'F1-Score'),
        'precision': (precisions, 'Precision'),
        'recall': (recalls, 'Recall'),
        'accuracy': (accuracies, 'Accuracy'),
        'auc_pr': (precisions * recalls, 'AUC-PR proxy (P*R)'),
        'auc_roc': (tprs - fprs, 'Youden J (TPR-FPR)'),
        'pr_balanced': (-np.abs(precisions - recalls), 'PR Balanced')
    }
    if optimize_metric not in metric_lookup:
        raise ValueError(
            f"Unknown optimize_metric: {optimize_metric}. Use one of {list(metric_lookup.keys())}")

    metric_values, metric_name = metric_lookup[optimize_metric]
    optimal_idx = int(np.argmax(metric_values))
    optimal_score = float(metric_values[optimal_idx])

    f1_idx = int(np.argmax(f1_scores))
    precision_idx = int(np.argmax(precisions))
    recall_idx = int(np.argmax(recalls))
    accuracy_idx = int(np.argmax(accuracies))
    auc_pr_idx = int(np.argmax(precisions * recalls))
    auc_roc_idx = int(np.argmax(tprs - fprs))

    f1_threshold = float(thresholds[f1_idx])
    precision_threshold = float(thresholds[precision_idx])
    recall_threshold = float(thresholds[recall_idx])
    accuracy_threshold = float(thresholds[accuracy_idx])
    auc_pr_threshold = float(thresholds[auc_pr_idx])
    auc_roc_threshold = float(thresholds[auc_roc_idx])
    
    print(f"\nDEBUG: {metric_name} statistics:")
    print(f"  Max {metric_name}: {optimal_score:.4f}")
    print(f"  Mean F1: {np.mean(f1_scores):.4f}")
    print(f"  Non-zero F1 values: {np.sum(np.array(f1_scores) > 0)}/100")
    
    if optimal_score == 0:
        print(f"\nERROR: Optimal {metric_name} is 0!")
        print("All precision and recall values are 0 across all thresholds.")
        print("Cannot optimize - returning None.")
        return None
    
    # Find threshold where precision and recall cross (first intersection)
    # Look for sign change in (precision - recall)
    pr_diff_signed = precisions - recalls
    sign_changes = np.where(np.diff(np.sign(pr_diff_signed)))[0]
    if len(sign_changes) > 0:
        # Use first crossing point
        pr_balanced_idx = sign_changes[0]
    else:
        # Fallback to minimum difference if no crossing found
        pr_diff = np.abs(precisions - recalls)
        pr_balanced_idx = np.argmin(pr_diff)
    pr_balanced_threshold = thresholds[pr_balanced_idx]
    
    # Find AUC-PR optimized threshold (maximize P * R, which approximates the curve area)
    pr_product = precisions * recalls
    auc_pr_idx = np.argmax(pr_product)
    auc_pr_threshold = thresholds[auc_pr_idx]
    
    print(f"\n{'='*80}")
    print("THRESHOLD ANALYSIS RESULTS")
    print(f"{'='*80}")
    print(f"F1 Threshold (max F1): {f1_threshold:.3f}")
    print(f"  F1-Score: {np.max(f1_scores):.4f}")
    print(f"  Precision: {precisions[f1_idx]:.4f}, Recall: {recalls[f1_idx]:.4f}, F1: {np.max(f1_scores):.4f}")
    print(f"\n{metric_name} Threshold: {thresholds[optimal_idx]:.3f}")
    print(f"  {metric_name}: {optimal_score:.4f}")
    print(f"  Precision: {precisions[optimal_idx]:.4f}, Recall: {recalls[optimal_idx]:.4f}, F1: {f1_scores[optimal_idx]:.4f}")
    print(f"\nAUC-PR Optimized (max P*R): {auc_pr_threshold:.3f}")
    print(f"  Precision: {precisions[auc_pr_idx]:.4f}, Recall: {recalls[auc_pr_idx]:.4f}, F1: {f1_scores[auc_pr_idx]:.4f}")
    print(f"\nAUC-ROC Optimized (max TPR-FPR): {auc_roc_threshold:.3f}")
    print(f"  TPR: {tprs[auc_roc_idx]:.4f}, FPR: {fprs[auc_roc_idx]:.4f}")
    print(f"\nPR Balanced Threshold (P≈R crossing): {pr_balanced_threshold:.3f}")
    print(f"  Precision: {precisions[pr_balanced_idx]:.4f}, Recall: {recalls[pr_balanced_idx]:.4f}, F1: {f1_scores[pr_balanced_idx]:.4f}")
    print(f"{'='*80}\n")
    
    # Create visualization plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Precision-Recall vs Threshold
    axes[0, 0].plot(thresholds, precisions, label='Precision', linewidth=2)
    axes[0, 0].plot(thresholds, recalls, label='Recall', linewidth=2)
    axes[0, 0].axvline(f1_threshold, color='r', linestyle='--', linewidth=1.5, label=f'F1-Score ({f1_threshold:.3f})')
    axes[0, 0].axvline(auc_pr_threshold, color='orange', linestyle='--', linewidth=1.5, label=f'AUC-PR ({auc_pr_threshold:.3f})')
    axes[0, 0].axvline(pr_balanced_threshold, color='purple', linestyle='-.', linewidth=1.5, label=f'PR Balanced ({pr_balanced_threshold:.3f})')
    axes[0, 0].set_xlabel('Confidence Threshold')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Precision and Recall vs Confidence Threshold')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Optimized Metric vs Threshold
    metric_plot_map = {
        'F1-Score': (f1_scores, 'F1 Score'),
        'Precision': (precisions, 'Precision'),
        'Recall': (recalls, 'Recall'),
        'Accuracy': (accuracies, 'Accuracy'),
        'AUC-PR proxy (P*R)': (pr_product, 'P * R (AUC-PR proxy)'),
        'Youden J (TPR-FPR)': (tprs - fprs, 'TPR - FPR (AUC-ROC proxy)'),
        'PR Balanced': (-np.abs(precisions - recalls), 'PR Balanced (|P-R|)')
    }
    metric_values, ylabel = metric_plot_map[metric_name]

    axes[0, 1].plot(thresholds, metric_values, linewidth=2, color='purple')
    axes[0, 1].axvline(thresholds[optimal_idx], color='r', linestyle='--', label=f'Max {metric_name} ({thresholds[optimal_idx]:.3f})')
    axes[0, 1].axhline(optimal_score, color='r', linestyle=':', alpha=0.5)
    axes[0, 1].set_xlabel('Confidence Threshold')
    axes[0, 1].set_ylabel(ylabel)
    axes[0, 1].set_title(f'{ylabel} vs Confidence Threshold (Max={optimal_score:.4f})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Precision-Recall Curve
    axes[1, 0].plot(recalls, precisions, linewidth=2, label='PR Curve')
    axes[1, 0].scatter([recalls[optimal_idx]], [precisions[optimal_idx]], 
                       c='red', s=150, zorder=5, marker='*', label=f'Optimized')
    axes[1, 0].scatter([recalls[auc_pr_idx]], [precisions[auc_pr_idx]], 
                       c='orange', s=150, zorder=5, marker='*', label=f'AUC-PR')
    axes[1, 0].scatter([recalls[auc_roc_idx]], [precisions[auc_roc_idx]], 
                       c='teal', s=140, zorder=5, marker='s', label=f'AUC-ROC')
    axes[1, 0].scatter([recalls[pr_balanced_idx]], [precisions[pr_balanced_idx]], 
                       c='purple', s=100, zorder=5, marker='D', label=f'PR Balanced')
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision-Recall Curve with Operating Points')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: ROC Curve
    axes[1, 1].plot(fprs, tprs, linewidth=2, color='teal', label=f'ROC (AUC={auc_roc:.3f})')
    axes[1, 1].scatter([fprs[auc_pr_idx]], [tprs[auc_pr_idx]], c='orange', s=120, marker='o', label='AUC-PR opt (P*R)')
    axes[1, 1].scatter([fprs[auc_roc_idx]], [tprs[auc_roc_idx]], c='red', s=140, marker='*', label='AUC-ROC opt')
    axes[1, 1].plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.5)
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title('ROC Curve')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 5: Confidence Distribution
    axes[1, 2].hist(all_confidences, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 2].axvline(f1_threshold, color='r', linestyle='--', 
                       label=f'F1-Score ({f1_threshold:.3f})', linewidth=2)
    axes[1, 2].axvline(auc_pr_threshold, color='orange', linestyle='--', 
                       label=f'AUC-PR ({auc_pr_threshold:.3f})', linewidth=2)
    axes[1, 2].axvline(auc_roc_threshold, color='teal', linestyle='--', 
                       label=f'AUC-ROC ({auc_roc_threshold:.3f})', linewidth=1.5)
    axes[1, 2].axvline(pr_balanced_threshold, color='purple', linestyle='-.', 
                       label=f'PR Balanced ({pr_balanced_threshold:.3f})', linewidth=1.5)
    axes[1, 2].set_xlabel('Confidence Score')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_title('Distribution of Prediction Confidence Scores')
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    # Plot 6: TP/FP Confidence Distribution (add to top right corner)
    axes[0, 2].hist([all_confidences[all_labels == 1]], bins=50, alpha=0.3, 
                    color='green', edgecolor='black', label='True Positives')
    axes[0, 2].hist([all_confidences[all_labels == 0]], bins=50, alpha=0.3, 
                    color='red', edgecolor='black', label='False Positives')
    axes[0, 2].set_xlabel('Confidence Score')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('TP vs FP Confidence Distribution')
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = plots_dir / 'threshold_optimization_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Threshold optimization plot saved to: {plot_path}")
    
    # Analyze low-confidence predictions for bias
    analyze_low_confidence_predictions(all_confidences, plots_dir)
    
    # Save threshold analysis results
    results = {
        'optimize_metric': optimize_metric,
        'f1_threshold': float(f1_threshold),
        'f1_score': float(np.max(f1_scores)),
        'f1_precision': float(precisions[f1_idx]),
        'f1_recall': float(recalls[f1_idx]),
        'f1_f1': float(f1_scores[f1_idx]),
        'precision_threshold': float(precision_threshold),
        'precision_precision': float(np.max(precisions)),
        'precision_recall': float(recalls[precision_idx]),
        'precision_f1': float(f1_scores[precision_idx]),
        'recall_threshold': float(recall_threshold),
        'recall_precision': float(precisions[recall_idx]),
        'recall_recall': float(np.max(recalls)),
        'recall_f1': float(f1_scores[recall_idx]),
        'accuracy_threshold': float(accuracy_threshold),
        'accuracy_score': float(np.max(accuracies)),
        'accuracy_precision': float(precisions[accuracy_idx]),
        'accuracy_recall': float(recalls[accuracy_idx]),
        'accuracy_f1': float(f1_scores[accuracy_idx]),
        'auc_pr_threshold': float(auc_pr_threshold),
        'auc_pr_precision': float(precisions[auc_pr_idx]),
        'auc_pr_recall': float(recalls[auc_pr_idx]),
        'auc_pr_f1': float(f1_scores[auc_pr_idx]),
        'auc_pr_curve_area': auc_pr,
        'auc_roc_threshold': float(auc_roc_threshold),
        'auc_roc_tpr': float(tprs[auc_roc_idx]),
        'auc_roc_fpr': float(fprs[auc_roc_idx]),
        'auc_roc_curve_area': auc_roc,
        'auc_roc_precision': float(precisions[auc_roc_idx]),
        'auc_roc_recall': float(recalls[auc_roc_idx]),
        'auc_roc_f1': float(f1_scores[auc_roc_idx]),
        'pr_balanced_threshold': float(pr_balanced_threshold),
        'pr_balanced_precision': float(precisions[pr_balanced_idx]),
        'pr_balanced_recall': float(recalls[pr_balanced_idx]),
        'pr_balanced_f1': float(f1_scores[pr_balanced_idx]),
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
            'f1_scores': f1_scores.tolist(),
            'accuracies': accuracies.tolist(),
            'tprs': tprs.tolist(),
            'fprs': fprs.tolist()
        }
    }
    
    results_path = output_dir / 'threshold_analysis_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Threshold analysis results saved to: {results_path}")
    
    # Create additional comparison plots
    create_threshold_comparison_plots(results, plots_dir)
    
    # Return both the results and the directories for external use
    results['_output_dir'] = str(output_dir)
    results['_plots_dir'] = str(plots_dir)
    results['_conf_matrix_dir'] = str(conf_matrix_dir)
    return results


def create_threshold_comparison_plots(threshold_analysis, plots_dir):
    """Create additional comparison plots for different threshold strategies."""
    print("\nCreating threshold strategy comparison plots...")
    
    # Build strategies dynamically based on available keys
    candidates = [
        ('f1', 'F1-Score'),
        ('auc_pr', 'AUC-PR'),
        ('auc_roc', 'AUC-ROC'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
        ('accuracy', 'Accuracy'),
        ('pr_balanced', 'PR Balanced')
    ]
    strategies = [name for name, _ in candidates if f'{name}_threshold' in threshold_analysis]
    strategy_labels = [label for name, label in candidates if f'{name}_threshold' in threshold_analysis]
    
    # Extract metrics for each strategy
    thresholds = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for strategy in strategies:
        thresholds.append(float(threshold_analysis.get(f'{strategy}_threshold', 0.0)))
        precisions.append(float(threshold_analysis.get(f'{strategy}_precision', 0.0)))
        recalls.append(float(threshold_analysis.get(f'{strategy}_recall', 0.0)))
        f1_scores.append(float(threshold_analysis.get(f'{strategy}_f1', 0.0)))
    
    colors = plt.cm.tab10(np.linspace(0, 1, max(3, len(strategies))))

    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Threshold values comparison
    x_pos = np.arange(len(strategies))
    axes[0, 0].bar(x_pos, thresholds, color=colors[:len(strategies)])
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(strategy_labels, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Confidence Threshold')
    axes[0, 0].set_title('Optimal Confidence Thresholds by Strategy')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for i, v in enumerate(thresholds):
        axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Metrics comparison (Precision, Recall, F1)
    x = np.arange(len(strategies))
    width = 0.25
    axes[0, 1].bar(x - width, precisions, width, label='Precision', color='#3498db')
    axes[0, 1].bar(x, recalls, width, label='Recall', color='#2ecc71')
    axes[0, 1].bar(x + width, f1_scores, width, label='F1-Score', color='#e74c3c')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(strategy_labels, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Metrics Comparison by Threshold Strategy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_ylim([0, 1.0])
    
    # Plot 3: PR Curve with all operating points
    all_thresholds = np.array(threshold_analysis['threshold_analysis']['thresholds'])
    all_precisions = np.array(threshold_analysis['threshold_analysis']['precisions'])
    all_recalls = np.array(threshold_analysis['threshold_analysis']['recalls'])
    
    axes[1, 0].plot(all_recalls, all_precisions, linewidth=2, color='gray', alpha=0.5, label='PR Curve')
    
    markers = ['*', 'o', 's', '^', 'D', 'P', 'X', 'v', '<', '>']
    for i, (strategy, label) in enumerate(zip(strategies, strategy_labels)):
        axes[1, 0].scatter([recalls[i]], [precisions[i]], 
                          c=[colors[i]], s=200, marker=markers[i % len(markers)], 
                          edgecolors='black', linewidths=2,
                          label=f'{label} (conf={thresholds[i]:.3f})', zorder=5)
    
    axes[1, 0].set_xlabel('Recall', fontsize=12)
    axes[1, 0].set_ylabel('Precision', fontsize=12)
    axes[1, 0].set_title('Precision-Recall Curve with All Threshold Strategies', fontsize=12)
    axes[1, 0].legend(loc='best', fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim([0, 1.0])
    axes[1, 0].set_ylim([0, 1.0])
    
    # Plot 4: F1 Score comparison with confidence values
    axes[1, 1].bar(x_pos, f1_scores, color=colors[:len(strategies)])
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(strategy_labels, rotation=45, ha='right')
    axes[1, 1].set_ylabel('F1-Score')
    axes[1, 1].set_title('F1-Score by Threshold Strategy')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_ylim([0, 1.0])
    # Add value labels with threshold info
    for i, (f1, thresh) in enumerate(zip(f1_scores, thresholds)):
        axes[1, 1].text(i, f1 + 0.02, f'F1: {f1:.3f}\nconf: {thresh:.3f}', 
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_path = plots_dir / 'threshold_strategies_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Threshold strategies comparison plot saved to: {plot_path}")
    
    return plot_path


def analyze_low_confidence_predictions(confidences, plots_dir):
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
    
    plot_path = plots_dir / 'confidence_distribution_analysis.png'
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
    
    stats_path = plots_dir / 'low_confidence_analysis.json'
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


def log_to_wandb(run_name, metrics_val_dict, metrics_test_dict, args_dict, 
                 threshold_configs=None, threshold_analysis=None, image_paths=None,
                 baseline_metrics_val=None, baseline_metrics_test=None, baseline_conf=None):
    """Log all metrics and optional images to Weights & Biases with clear naming.
    
    Naming convention: {split}/{strategy}_conf/{metric_name}
    Example: val/f1_conf/precision means validation precision using F1-optimized confidence threshold
    """
    run = wandb.init(
        project="DeTect-BMMS-Inference",
        name=run_name,
        config=args_dict,
        tags=["inference", "threshold-optimization"]
    )
    
    strategy_order = list(threshold_configs.keys()) if threshold_configs else ['f1', 'auc_pr', 'auc_roc', 'precision', 'recall', 'accuracy', 'balanced']
    strategy_display = {
        'f1': 'F1-Score',
        'auc_pr': 'AUC-PR',
        'auc_roc': 'AUC-ROC',
        'precision': 'Precision',
        'recall': 'Recall',
        'accuracy': 'Accuracy',
        'pr_balanced': 'PR Balanced'
    }
    
    print("\n" + "="*80)
    print("LOGGING TO WANDB")
    print("="*80)
    
    # Log baseline mAP for reference
    if baseline_conf is not None and (baseline_metrics_val or baseline_metrics_test):
        print(f"\n📊 Logging Baseline mAP (conf={baseline_conf:.4f}):")
        baseline_log = {}
        # baseline_log['baseline_mAP/confidence'] = baseline_conf
        
        if baseline_metrics_val:
            baseline_log['baseline_mAP/val/mAP50'] = float(baseline_metrics_val.box.map50) if hasattr(baseline_metrics_val.box, 'map50') else None
            baseline_log['baseline_mAP/val/mAP50-95'] = float(baseline_metrics_val.box.map) if hasattr(baseline_metrics_val.box, 'map') else None
            print(f"  Val - mAP50: {baseline_log['baseline_mAP/val/mAP50']:.4f}, mAP50-95: {baseline_log['baseline_mAP/val/mAP50-95']:.4f}")
        
        if baseline_metrics_test:
            baseline_log['baseline_mAP/test/mAP50'] = float(baseline_metrics_test.box.map50) if hasattr(baseline_metrics_test.box, 'map50') else None
            baseline_log['baseline_mAP/test/mAP50-95'] = float(baseline_metrics_test.box.map) if hasattr(baseline_metrics_test.box, 'map') else None
            print(f"  Test - mAP50: {baseline_log['baseline_mAP/test/mAP50']:.4f}, mAP50-95: {baseline_log['baseline_mAP/test/mAP50-95']:.4f}")
        
        wandb.log(baseline_log)
    
    # Log threshold values from optimization
    if threshold_configs:
        print("\n📊 Logging Optimized Confidence Thresholds:")
        threshold_log = {}
        for strategy_name, threshold_value in threshold_configs.items():
            threshold_log[f'thresholds/{strategy_name}_optimized'] = threshold_value
            print(f"  thresholds/{strategy_name}_optimized = {threshold_value:.4f}")
        wandb.log(threshold_log)
    
    # Log threshold analysis metrics if available
    if threshold_analysis:
        print("\n📊 Logging Threshold Analysis Metrics:")
        analysis_log = {}
        for strategy_name in strategy_order:
            if f'{strategy_name}_threshold' in threshold_analysis:
                prefix = f'threshold_analysis/{strategy_name}'
                analysis_log[f'{prefix}/threshold'] = threshold_analysis[f'{strategy_name}_threshold']
                analysis_log[f'{prefix}/precision'] = threshold_analysis[f'{strategy_name}_precision']
                analysis_log[f'{prefix}/recall'] = threshold_analysis[f'{strategy_name}_recall']
                analysis_log[f'{prefix}/f1'] = threshold_analysis[f'{strategy_name}_f1']
                display_name = strategy_display.get(strategy_name, strategy_name)
                
                print(f"  {prefix} ({display_name}): conf={threshold_analysis[f'{strategy_name}_threshold']:.4f}, "
                      f"P={threshold_analysis[f'{strategy_name}_precision']:.4f}, "
                      f"R={threshold_analysis[f'{strategy_name}_recall']:.4f}, "
                      f"F1={threshold_analysis[f'{strategy_name}_f1']:.4f}")
        analysis_log['threshold_analysis/auc_pr_curve_area'] = threshold_analysis.get('auc_pr_curve_area')
        analysis_log['threshold_analysis/auc_roc_curve_area'] = threshold_analysis.get('auc_roc_curve_area')
        
        wandb.log(analysis_log)
    
    # Log validation metrics for all threshold strategies
    print("\n📊 Logging Validation Metrics:")
    val_log = {}
    for strategy_name in strategy_order:
        if strategy_name not in metrics_val_dict:
            continue
        metrics = metrics_val_dict[strategy_name]
        prefix = f'val/{strategy_name}_conf'
        
        # Log all key metrics with clear naming: val/{strategy}/{metric}
        val_log[f'{prefix}/precision'] = float(metrics.box.mp)  # mean precision
        val_log[f'{prefix}/recall'] = float(metrics.box.mr)     # mean recall
        val_log[f'{prefix}/f1'] = float(2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr + 1e-6))
        val_log[f'{prefix}/mAP50'] = float(metrics.box.map50)
        val_log[f'{prefix}/mAP50-95'] = float(metrics.box.map)
        
        display_name = strategy_display.get(strategy_name, strategy_name)
        print(f"  {display_name} threshold:")
        print(f"    val/{strategy_name}/precision: {metrics.box.mp:.4f}")
        print(f"    val/{strategy_name}/recall: {metrics.box.mr:.4f}")
        print(f"    val/{strategy_name}/f1: {val_log[f'{prefix}/f1']:.4f}")
        print(f"    val/{strategy_name}/mAP50: {metrics.box.map50:.4f}")
        print(f"    val/{strategy_name}/mAP50-95: {metrics.box.map:.4f}")
    
    if val_log:
        wandb.log(val_log)
    
    # Log test metrics for all threshold strategies
    if metrics_test_dict:
        print("\n📊 Logging Test Metrics:")
        test_log = {}
        for strategy_name in strategy_order:
            if strategy_name not in metrics_test_dict:
                continue
            metrics = metrics_test_dict[strategy_name]
            prefix = f'test/{strategy_name}'
            
            # Log all key metrics with clear naming: test/{strategy}/{metric}
            test_log[f'{prefix}/precision'] = float(metrics.box.mp)
            test_log[f'{prefix}/recall'] = float(metrics.box.mr)
            test_log[f'{prefix}/f1'] = float(2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr + 1e-6))
            test_log[f'{prefix}/mAP50'] = float(metrics.box.map50)
            test_log[f'{prefix}/mAP50-95'] = float(metrics.box.map)
            
            display_name = strategy_display.get(strategy_name, strategy_name)
            print(f"  {display_name} threshold:")
            print(f"    test/{strategy_name}/precision: {metrics.box.mp:.4f}")
            print(f"    test/{strategy_name}/recall: {metrics.box.mr:.4f}")
            print(f"    test/{strategy_name}/f1: {test_log[f'{prefix}/f1']:.4f}")
            print(f"    test/{strategy_name}/mAP50: {metrics.box.map50:.4f}")
            print(f"    test/{strategy_name}/mAP50-95: {metrics.box.map:.4f}")
        
        if test_log:
            wandb.log(test_log)
    
    print("\n📊 All metrics logged to wandb")
    
    # Create comprehensive summary tables
    print("\n📊 Creating Summary Tables:")
    
    # Table 1: Detailed metrics comparison
    summary_data = []
    for strategy_name in strategy_order:
        if strategy_name not in metrics_val_dict:
            continue
        metrics = metrics_val_dict[strategy_name]
        conf_thresh = threshold_configs.get(strategy_name, 0.0) if threshold_configs else 0.0
        f1 = 2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr + 1e-6)
        summary_data.append([
            "Validation",
            strategy_display.get(strategy_name, strategy_name),
            round(conf_thresh, 4),
            round(metrics.box.mp, 4),
            round(metrics.box.mr, 4),
            round(f1, 4),
            round(metrics.box.map50, 4),
            round(metrics.box.map, 4)
        ])
    
    for strategy_name in strategy_order:
        if strategy_name not in metrics_test_dict:
            continue
        metrics = metrics_test_dict[strategy_name]
        conf_thresh = threshold_configs.get(strategy_name, 0.0) if threshold_configs else 0.0
        f1 = 2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr + 1e-6)
        summary_data.append([
            "Test",
            strategy_display.get(strategy_name, strategy_name),
            round(conf_thresh, 4),
            round(metrics.box.mp, 4),
            round(metrics.box.mr, 4),
            round(f1, 4),
            round(metrics.box.map50, 4),
            round(metrics.box.map, 4)
        ])
    
    table = wandb.Table(
        columns=["Split", "Strategy", "Conf Threshold", "Precision", "Recall", "F1-Score", "mAP50", "mAP50-95"],
        data=summary_data
    )
    wandb.log({"metrics_summary_table": table})
    print("  ✓ Metrics summary table created")
    
    # Table 2: Threshold optimization results
    if threshold_analysis:
        threshold_data = []
        for strategy_name in strategy_order:
            if f'{strategy_name}_threshold' in threshold_analysis:
                threshold_data.append([
                    strategy_display.get(strategy_name, strategy_name),
                    round(threshold_analysis[f'{strategy_name}_threshold'], 4),
                    round(threshold_analysis[f'{strategy_name}_precision'], 4),
                    round(threshold_analysis[f'{strategy_name}_recall'], 4),
                    round(threshold_analysis[f'{strategy_name}_f1'], 4)
                ])
        
        threshold_table = wandb.Table(
            columns=["Strategy", "Optimal Conf", "Precision@Conf", "Recall@Conf", "F1@Conf"],
            data=threshold_data
        )
        wandb.log({"threshold_optimization_table": threshold_table})
        print("  ✓ Threshold optimization table created")

    # Optional image logging with organized panels
    if image_paths:
        print("\n📊 Logging Analysis Plots:")
        for img_name, img_path in image_paths.items():
            if Path(img_path).exists():
                # Organize images into different WandB panels
                if 'confusion_matrix' in img_name:
                    log_key = f"confusion_matrices/{img_name}"
                elif 'pr_curve' in img_name or 'f1_curve' in img_name:
                    log_key = f"performance_curves/{img_name}"
                elif img_name in ['threshold_optimization', 'confidence_distribution', 'threshold_strategies_comparison']:
                    log_key = f"threshold_analysis/{img_name}"
                else:
                    log_key = f"plots/{img_name}"
                
                wandb.log({log_key: wandb.Image(str(img_path))})
                print(f"  ✓ Logged: {log_key}")
    
    print("\n" + "="*80)
    wandb.finish()
    print(f"✓ WandB run finished: {run.url}")
    print("="*80 + "\n")


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

    parser.add_argument('--log-images', action='store_true',
                        help='Log key result plots (threshold, confidence, val/test plots) to wandb')

    parser.add_argument('--save-plots', type=int, default=0,
                        help='Number of sample visualizations (pred vs GT) to save for debugging')
    
    parser.add_argument('--optimize-metric', type=str, default='f1',
                        choices=['f1', 'precision', 'recall', 'accuracy', 'auc_pr', 'auc_roc', 'pr_balanced'],
                        help='Metric to optimize for threshold selection (f1/precision/recall/accuracy/auc_pr/auc_roc/pr_balanced)')
    
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
        iou_threshold=args.iou,
        save_plots=args.save_plots,
        optimize_metric=args.optimize_metric
    )
    
    if threshold_analysis:
        f1_threshold = threshold_analysis['f1_threshold']
        auc_pr_threshold = threshold_analysis['auc_pr_threshold']
        auc_roc_threshold = threshold_analysis.get('auc_roc_threshold', auc_pr_threshold)
        precision_threshold = threshold_analysis['precision_threshold']
        recall_threshold = threshold_analysis['recall_threshold']
        accuracy_threshold = threshold_analysis.get('accuracy_threshold', f1_threshold)
        pr_balanced_threshold = threshold_analysis['pr_balanced_threshold']
        
        # Extract directory paths from results
        analysis_output_dir = Path(threshold_analysis.get('_output_dir', str(analysis_output_dir)))
        plots_dir = Path(threshold_analysis.get('_plots_dir', str(analysis_output_dir / 'plots')))
        conf_matrix_dir = Path(threshold_analysis.get('_conf_matrix_dir', str(analysis_output_dir / 'conf_matrix')))
        print(f"\n{'='*80}")
        print(f"✓ THRESHOLDS FOUND:")
        print(f"  F1-Score: {f1_threshold:.3f}")
        print(f"  AUC-PR optimized: {auc_pr_threshold:.3f}")
        print(f"  AUC-ROC optimized: {auc_roc_threshold:.3f}")
        print(f"  Precision optimized: {precision_threshold:.3f}")
        print(f"  Recall optimized: {recall_threshold:.3f}")
        print(f"  Accuracy optimized: {accuracy_threshold:.3f}")
        print(f"  PR Balanced (P≈R crossing): {pr_balanced_threshold:.3f}")
        print(f"{'='*80}\n")
    else:
        print("Warning: Could not determine optimal thresholds, using defaults")
        f1_threshold = 0.25
        auc_pr_threshold = 0.25
        auc_roc_threshold = 0.25
        precision_threshold = 0.25
        recall_threshold = 0.25
        accuracy_threshold = 0.25
        pr_balanced_threshold = 0.25
    
    # PHASE 3: Validation with ALL optimized thresholds
    print(f"\n{'='*80}")
    print(f"PHASE 3: VALIDATION WITH ALL OPTIMIZED THRESHOLDS")
    print(f"{'='*80}\n")
    
    threshold_configs = {
        'f1': f1_threshold,
        'auc_pr': auc_pr_threshold,
        'auc_roc': auc_roc_threshold,
        'precision': precision_threshold,
        'recall': recall_threshold,
        'accuracy': accuracy_threshold,
        'pr_balanced': pr_balanced_threshold
    }
    
    metrics_val_dict = {}
    for strategy_name, threshold_value in threshold_configs.items():
        print(f"\n--- Validation with {strategy_name.upper()} threshold ({threshold_value:.3f}) ---")
        metrics = run_inference(
            model=model,
            data_yaml=args.data,
            split='val',
            project=args.project,
            exp_name=f"{args.name}_{strategy_name}",
            single_cls=args.single_cls,
            conf=threshold_value,
            iou=args.iou,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            max_det=args.max_det,
            augment=False,
        )
        metrics_val_dict[strategy_name] = metrics
        
        val_path = Path(args.project) / f'inference/val/{args.name}_{strategy_name}/metrics.txt'
        save_metrics(metrics, val_path)
    
    print("\n" + "="*80)
    print("VALIDATION RESULTS COMPARISON (ALL THRESHOLDS)")
    print("="*80)
    for strategy_name, metrics in metrics_val_dict.items():
        threshold_value = threshold_configs[strategy_name]
        print(f"\n{strategy_name.upper()} Threshold: {threshold_value:.3f}")
        print(f"  Precision: {float(metrics.box.mp):.4f}" if hasattr(metrics.box, 'mp') else "  Precision: N/A")
        print(f"  Recall: {float(metrics.box.mr):.4f}" if hasattr(metrics.box, 'mr') else "  Recall: N/A")
        print(f"  F1-Score: {2 * float(metrics.box.mp) * float(metrics.box.mr) / (float(metrics.box.mp) + float(metrics.box.mr) + 1e-10):.4f}" if hasattr(metrics.box, 'mp') else "  F1-Score: N/A")
        print(f"  mAP50: {float(metrics.box.map50):.4f}" if hasattr(metrics.box, 'map50') else "  mAP50: N/A")
        print(f"  mAP50-95: {float(metrics.box.map):.4f}" if hasattr(metrics.box, 'map') else "  mAP50-95: N/A")
    print("="*80 + "\n")
    
    # PHASE 4: Test with ALL optimized thresholds
    metrics_test_dict = {}
    metrics_test_lowconf = None
    if not args.val_only:
        print(f"\n{'='*80}")
        print(f"PHASE 4: TEST WITH ALL OPTIMIZED THRESHOLDS")
        print(f"{'='*80}\n")
        
        # First run low-confidence test for baseline
        print("Running low-confidence test for baseline mAP...")
        metrics_test_lowconf = run_inference(
            model=model,
            data_yaml=args.data,
            split='test',
            project=args.project,
            exp_name=f"{args.name}_test_lowconf",
            single_cls=args.single_cls,
            conf=args.initial_conf,
            iou=args.iou,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            max_det=args.max_det,
            augment=False,
        )
        print("✓ Low-confidence test baseline completed\n")
        
        for strategy_name, threshold_value in threshold_configs.items():
            print(f"\n--- Test with {strategy_name.upper()} threshold ({threshold_value:.3f}) ---")
            metrics = run_inference(
                model=model,
                data_yaml=args.data,
                split='test',
                project=args.project,
                exp_name=f"{args.name}_{strategy_name}",
                single_cls=args.single_cls,
                conf=threshold_value,
                iou=args.iou,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                max_det=args.max_det,
                augment=False
            )
            metrics_test_dict[strategy_name] = metrics
            
            test_path = Path(args.project) / f'inference/test/{args.name}_{strategy_name}/metrics.txt'
            save_metrics(metrics, test_path)
        
        print("\n" + "="*80)
        print("TEST RESULTS COMPARISON (ALL THRESHOLDS)")
        print("="*80)
        for strategy_name, metrics in metrics_test_dict.items():
            threshold_value = threshold_configs[strategy_name]
            print(f"\n{strategy_name.upper()} Threshold: {threshold_value:.3f}")
            print(f"  Precision: {float(metrics.box.mp):.4f}" if hasattr(metrics.box, 'mp') else "  Precision: N/A")
            print(f"  Recall: {float(metrics.box.mr):.4f}" if hasattr(metrics.box, 'mr') else "  Recall: N/A")
            print(f"  F1-Score: {2 * float(metrics.box.mp) * float(metrics.box.mr) / (float(metrics.box.mp) + float(metrics.box.mr) + 1e-10):.4f}" if hasattr(metrics.box, 'mp') else "  F1-Score: N/A")
            print(f"  mAP50: {float(metrics.box.map50):.4f}" if hasattr(metrics.box, 'map50') else "  mAP50: N/A")
            print(f"  mAP50-95: {float(metrics.box.map):.4f}" if hasattr(metrics.box, 'map') else "  mAP50-95: N/A")
        print("="*80 + "\n")
    
    # Save concise mAP summaries for quick reference
    map_summary = {}
    
    # Add baseline (low-confidence) mAP as reference point
    print(f"\n{'='*80}")
    print("BASELINE mAP (Low-confidence conf={:.4f}):".format(args.initial_conf))
    print(f"{'='*80}")
    
    # Validation baseline
    if metrics_val_lowconf:
        baseline_val_map50 = float(metrics_val_lowconf.box.map50) if hasattr(metrics_val_lowconf.box, 'map50') else None
        baseline_val_map50_95 = float(metrics_val_lowconf.box.map) if hasattr(metrics_val_lowconf.box, 'map') else None
        map_summary['baseline/val'] = {
            'confidence': args.initial_conf,
            'mAP50': baseline_val_map50,
            'mAP50_95': baseline_val_map50_95
        }
        print(f"Validation Baseline:")
        print(f"  mAP50: {baseline_val_map50:.4f}" if baseline_val_map50 else "  mAP50: N/A")
        print(f"  mAP50-95: {baseline_val_map50_95:.4f}" if baseline_val_map50_95 else "  mAP50-95: N/A")
    
    # Test baseline
    if metrics_test_lowconf:
        baseline_test_map50 = float(metrics_test_lowconf.box.map50) if hasattr(metrics_test_lowconf.box, 'map50') else None
        baseline_test_map50_95 = float(metrics_test_lowconf.box.map) if hasattr(metrics_test_lowconf.box, 'map') else None
        map_summary['baseline/test'] = {
            'confidence': args.initial_conf,
            'mAP50': baseline_test_map50,
            'mAP50_95': baseline_test_map50_95
        }
        print(f"Test Baseline:")
        print(f"  mAP50: {baseline_test_map50:.4f}" if baseline_test_map50 else "  mAP50: N/A")
        print(f"  mAP50-95: {baseline_test_map50_95:.4f}" if baseline_test_map50_95 else "  mAP50-95: N/A")
    
    print(f"{'='*80}\n")
    
    for strategy_name, metrics in metrics_val_dict.items():
        map_summary[f'val/{strategy_name}'] = {
            'mAP50': float(metrics.box.map50) if hasattr(metrics.box, 'map50') else None,
            'mAP50_95': float(metrics.box.map) if hasattr(metrics.box, 'map') else None
        }
    for strategy_name, metrics in metrics_test_dict.items():
        map_summary[f'test/{strategy_name}'] = {
            'mAP50': float(metrics.box.map50) if hasattr(metrics.box, 'map50') else None,
            'mAP50_95': float(metrics.box.map) if hasattr(metrics.box, 'map') else None
        }
    map_summary_path = analysis_output_dir / 'map_summary.json'
    map_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(map_summary_path, 'w') as f:
        json.dump(map_summary, f, indent=2)
    print(f"mAP summary saved to: {map_summary_path}")
    
    # Log to wandb
    if not args.no_wandb:
        # Log baseline mAP for reference
        if metrics_val_lowconf and 'baseline' in map_summary:
            baseline_log = {
                f'baseline_mAP/conf': args.initial_conf,
                f'baseline_mAP/mAP50': map_summary['baseline']['mAP50'],
                f'baseline_mAP/mAP50-95': map_summary['baseline']['mAP50_95']
            }
            print(f"📊 Logging Baseline mAP (conf={args.initial_conf:.4f}):")
            print(f"  baseline_mAP/mAP50: {map_summary['baseline']['mAP50']:.4f}")
            print(f"  baseline_mAP/mAP50-95: {map_summary['baseline']['mAP50_95']:.4f}")
        
        # Collect image paths if log_images is enabled
        image_paths = None
        if args.log_images:
            image_paths = {}
            
            # Threshold optimization plots
            thresh_plot = plots_dir / 'threshold_optimization_analysis.png'
            if thresh_plot.exists():
                image_paths['threshold_optimization'] = str(thresh_plot)
            
            conf_dist_plot = plots_dir / 'confidence_distribution_analysis.png'
            if conf_dist_plot.exists():
                image_paths['confidence_distribution'] = str(conf_dist_plot)
            
            # New comparison plot
            comparison_plot = plots_dir / 'threshold_strategies_comparison.png'
            if comparison_plot.exists():
                image_paths['threshold_strategies_comparison'] = str(comparison_plot)
            
            # Validation plots for each threshold
            for strategy_name in threshold_configs.keys():
                val_dir = Path(args.project) / f'inference/val/{args.name}_{strategy_name}'
                
                # Confusion matrix
                cm_plot = val_dir / 'confusion_matrix.png'
                if cm_plot.exists():
                    image_paths[f'val_{strategy_name}_confusion_matrix'] = str(cm_plot)
                
                # PR curve
                pr_plot = val_dir / 'PR_curve.png'
                if pr_plot.exists():
                    image_paths[f'val_{strategy_name}_pr_curve'] = str(pr_plot)
                
                # F1 curve
                f1_plot = val_dir / 'F1_curve.png'
                if f1_plot.exists():
                    image_paths[f'val_{strategy_name}_f1_curve'] = str(f1_plot)
            
            # Test plots for each threshold (if not val-only)
            if not args.val_only:
                for strategy_name in threshold_configs.keys():
                    test_dir = Path(args.project) / f'inference/test/{args.name}_{strategy_name}'
                    
                    # Confusion matrix
                    cm_plot = test_dir / 'confusion_matrix.png'
                    if cm_plot.exists():
                        image_paths[f'test_{strategy_name}_confusion_matrix'] = str(cm_plot)
                    
                    # PR curve
                    pr_plot = test_dir / 'PR_curve.png'
                    if pr_plot.exists():
                        image_paths[f'test_{strategy_name}_pr_curve'] = str(pr_plot)
                    
                    # F1 curve
                    f1_plot = test_dir / 'F1_curve.png'
                    if f1_plot.exists():
                        image_paths[f'test_{strategy_name}_f1_curve'] = str(f1_plot)
        
        # Prepare config for wandb
        wandb_config = {
            'model_path': args.model_path,
            'experiment_name': args.name,
            'data_yaml': args.data,
            'single_cls': args.single_cls,
            'initial_conf': args.initial_conf,
            'iou_threshold': args.iou,
            'optimize_metric': args.optimize_metric,
            'imgsz': args.imgsz,
            'batch_size': args.batch,
            'device': args.device if args.device else 'auto',
            'max_det': args.max_det,
            'val_only': args.val_only,
            'thresholds': threshold_configs
        }
        
        log_to_wandb(
            run_name=run_name,
            metrics_val_dict=metrics_val_dict,
            metrics_test_dict=metrics_test_dict,
            args_dict=wandb_config,
            threshold_configs=threshold_configs,
            threshold_analysis=threshold_analysis,
            image_paths=image_paths,
            baseline_metrics_val=metrics_val_lowconf,
            baseline_metrics_test=metrics_test_lowconf,
            baseline_conf=args.initial_conf
        )
    
    print("\n" + "="*80)
    print("✓ INFERENCE COMPLETED SUCCESSFULLY!")
    print(f"✓ F1-Score threshold: {f1_threshold:.3f}")
    print(f"✓ AUC-PR threshold: {auc_pr_threshold:.3f}")
    print(f"✓ AUC-ROC threshold: {auc_roc_threshold:.3f}")
    print(f"✓ Precision threshold: {precision_threshold:.3f}")
    print(f"✓ Recall threshold: {recall_threshold:.3f}")
    print(f"✓ Accuracy threshold: {accuracy_threshold:.3f}")
    print(f"✓ PR Balanced threshold (P≈R crossing): {pr_balanced_threshold:.3f}")
    print(f"✓ Analysis saved to: {analysis_output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
