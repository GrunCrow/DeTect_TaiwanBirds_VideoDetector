"""
Detection Analysis Utilities for Object Detection Models

This module provides comprehensive analysis tools for evaluating object detection
models by computing True Positives, False Positives, False Negatives, and various
performance metrics.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Union
import cv2


def plot_training_loss_curves(results_df, RESULTS_DIR):
    if results_df is not None:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        
        # Box Loss
        ax = axes[0]
        ax.plot(results_df.index, results_df['train/box_loss'], label='Train Box Loss', marker='o', markersize=3)
        ax.plot(results_df.index, results_df['val/box_loss'], label='Val Box Loss', marker='s', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Box Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Class Loss
        ax = axes[1]
        ax.plot(results_df.index, results_df['train/cls_loss'], label='Train Class Loss', marker='o', markersize=3)
        ax.plot(results_df.index, results_df['val/cls_loss'], label='Val Class Loss', marker='s', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Classification Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # DFL Los
        ax = axes[2]
        ax.plot(results_df.index, results_df['train/dfl_loss'], label='Train DFL Loss', marker='o', markersize=3)
        ax.plot(results_df.index, results_df['val/dfl_loss'], label='Val DFL Loss', marker='s', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('DFL Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # # Total Loss
        # ax = axes[3]
        # if 'train/loss' in results_df.columns:
        #     ax.plot(results_df.index, results_df['train/loss'], label='Train Loss', marker='o', markersize=3)
        # if 'val/loss' in results_df.columns:
        #     ax.plot(results_df.index, results_df['val/loss'], label='Val Loss', marker='s', markersize=3)
        # ax.set_xlabel('Epoch')
        # ax.set_ylabel('Loss')
        # ax.set_title('Total Loss')
        # ax.legend()
        # ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'curves_loss.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("✓ Loss plots generated and saved to 'curves_loss.png'")
    else:
        print("Cannot generate loss plots - results not available")

def plot_training_metrics_curves(results_df, RESULTS_DIR):
    if results_df is not None:
        fig, axes = plt.subplots(1, 2, figsize=(20, 5))
        
        # mAP@0.5
        ax = axes[0]
        ax.plot(results_df.index, results_df['metrics/mAP50(B)'], label='mAP@0.5', marker='o', markersize=4, linewidth=2)
        ax.plot(results_df.index, results_df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', marker='s', markersize=4, linewidth=2, color='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP')
        ax.set_title('mAP vs Epochs')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim([0, 1])

        # Precision-Recall
        ax = axes[1]
        ax.plot(results_df.index, results_df['metrics/precision(B)'], label='Precision (Val)', marker='o', markersize=4, linewidth=2)
        ax.plot(results_df.index, results_df['metrics/recall(B)'], label='Recall (Val)', marker='s', markersize=4, linewidth=2, color='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('')
        ax.set_title('Precisión-Recall vs Epochs')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'curves_map-pr.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("✓ mAP-PR plots saved")
        
        # Print max values
        print(f"\nBest mAP@0.5: {results_df['metrics/mAP50(B)'].max():.4f} at epoch {results_df['metrics/mAP50(B)'].idxmax()}")
        print(f"Best mAP@0.5:0.95: {results_df['metrics/mAP50-95(B)'].max():.4f} at epoch {results_df['metrics/mAP50-95(B)'].idxmax()}")
        # Print max values
        print(f"\nBest Precision: {results_df['metrics/precision(B)'].max():.4f} at epoch {results_df['metrics/precision(B)'].idxmax()}")
        print(f"Best Recall: {results_df['metrics/recall(B)'].max():.4f} at epoch {results_df['metrics/recall(B)'].idxmax()}")
    else:
        print("Plot generation failed - results not available")


def run_stream_predictions_to_json(
    model,
    source,
    output_path,
    conf,
    imgsz,
    device,
    batch=1,
    save=True,
    stream=True,
    verbose=False,
    log_every=500,
    ):
    results = model.predict(
        source=source,
        conf=conf,
        imgsz=imgsz,
        batch=batch,
        device=device,
        save=save,
        stream=stream,
        verbose=verbose,
    )

    predictions = []
    count = 0
    print("Processing predictions...")
    for result in results:
        pred_json = json.loads(result.to_json())
        # Store source image path alongside predictions to keep alignment
        predictions.append({
            "image": result.path,
            "predictions": pred_json,
        })
        count += 1
        if log_every and count % log_every == 0:
            print(f"  Processed {count} images...")

    with open(output_path, "w") as f:
        json.dump(predictions, f)

    print(f"Saved {count} predictions to: {output_path}")
    return count


def load_predictions_grouped(predictions_path):
    if not predictions_path.exists():
        print(f"✗ Archivo de predicciones no encontrado: {predictions_path}")
        return {}

    with open(predictions_path, "r") as f:
        predictions_list = json.load(f)

    grouped = {}
    if not predictions_list:
        return grouped

    first_item = predictions_list[0]
    if isinstance(first_item, dict) and "predictions" in first_item:
        for pred_item in predictions_list:
            img_id = pred_item.get("image") or "unknown"
            grouped[img_id] = {"detections": [], "boxes": []}

            pred = pred_item.get("predictions", {})
            if isinstance(pred, list):
                grouped[img_id]["detections"].extend(pred)
            elif "results" in pred and isinstance(pred["results"], list):
                grouped[img_id]["detections"].extend(pred["results"])
            elif "boxes" in pred:
                grouped[img_id]["boxes"].append(pred)
            else:
                grouped[img_id]["detections"].append(pred)
    elif isinstance(first_item, dict):
        for pred in predictions_list:
            img_id = pred.get("image_id") or pred.get("image_name") or pred.get("filename") or "unknown"
            if img_id not in grouped:
                grouped[img_id] = {"boxes": [], "detections": []}

            if "results" in pred and isinstance(pred["results"], list):
                grouped[img_id]["detections"].extend(pred["results"])
            elif "boxes" in pred:
                grouped[img_id]["boxes"].append(pred)
            else:
                grouped[img_id]["detections"].append(pred)
    elif isinstance(first_item, list):
        for idx, detections in enumerate(predictions_list):
            if isinstance(detections, list):
                img_id = f"image_{idx}"
                grouped[img_id] = {"detections": detections, "boxes": []}
    else:
        print(f"⚠ Unexpected predictions list format: {type(first_item)}")
        print(f"First element sample: {str(first_item)[:200]}")

    return grouped


def basic_statistics(predictions):
    # Basic statistics about detections
    if predictions:
        # Count total detections from flattened structure
        total_detections = 0
        for img_id, data in predictions.items():
            if isinstance(data, dict):
                total_detections += len(data.get('detections', []))
                total_detections += len(data.get('boxes', []))
            elif isinstance(data, list):
                total_detections += len(data)
        
        images_with_detections = sum(1 for data in predictions.values() 
                                    if (isinstance(data, dict) and (data.get('detections') or data.get('boxes'))) 
                                    or (isinstance(data, list) and len(data) > 0))
        
        print(f"\n--- Statistics ---")
        print(f"Total images: {len(predictions)}")
        print(f"Images with detections: {images_with_detections}")
        print(f"Images without detections: {len(predictions) - images_with_detections}")
        print(f"Total detections: {total_detections}")
        print(f"Average detections per image: {total_detections / len(predictions):.2f}")
        
        # Analysis of confidence scores
        all_confidences = []
        for data in predictions.values():
            if isinstance(data, dict):
                detections = data.get('detections', []) + data.get('boxes', [])
            else:
                detections = data if isinstance(data, list) else []
            
            for pred in detections:
                if isinstance(pred, dict):
                    # Handle different possible key names for confidence
                    conf = pred.get('confidence') or pred.get('score') or pred.get('conf')
                    if conf is not None:
                        all_confidences.append(conf)
        
        if all_confidences:
            print(f"\n--- Confidence Scores ---")
            print(f"Min: {min(all_confidences):.4f}")
            print(f"Max: {max(all_confidences):.4f}")
            print(f"Mean: {np.mean(all_confidences):.4f}")
            print(f"Median: {np.median(all_confidences):.4f}")
            print(f"Std Dev: {np.std(all_confidences):.4f}")
        else:
            print("\n⚠ Could not extract confidence scores")
            print("Checking structure of the first prediction:")
            if predictions:
                first_img_data = next(iter(predictions.values()))
                if isinstance(first_img_data, dict):
                    first_detection = (first_img_data.get('detections', first_img_data.get('boxes', [None])))[0] if first_img_data.get('detections') or first_img_data.get('boxes') else None
                else:
                    first_detection = first_img_data[0] if first_img_data else None
                
                if first_detection and isinstance(first_detection, dict):
                    print(f"Available keys: {first_detection.keys()}")
    
    if all_confidences:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        ax = axes[0]
        ax.hist(all_confidences, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(np.mean(all_confidences), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_confidences):.3f}')
        ax.axvline(np.median(all_confidences), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(all_confidences):.3f}')
        ax.axvline(0.5, color='orange', linestyle='--', linewidth=2, label='Threshold: 0.5')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Confidence Score Distribution (Validation)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Cumulative Distribution
        ax = axes[1]
        sorted_conf = np.sort(all_confidences)
        cumulative = np.arange(1, len(sorted_conf) + 1) / len(sorted_conf)
        ax.plot(sorted_conf, cumulative, linewidth=2)
        ax.axvline(0.5, color='orange', linestyle='--', linewidth=2, label='Threshold: 0.5')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Cumulative Distribution')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'val_confidence_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("✓ Confidence distribution plots saved")
        
        # Count predictions by confidence range
        print("\n--- Predictions by Confidence Range ---")
        ranges = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
        for low, high in ranges:
            count = sum(1 for c in all_confidences if low <= c < high)
            pct = count / len(all_confidences) * 100
            print(f"{low:.2f} - {high:.2f}: {count} predictions ({pct:.1f}%)")


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: Bounding box in format [x1, y1, x2, y2]
        box2: Bounding box in format [x1, y1, x2, y2]
        
    Returns:
        IoU score between 0 and 1
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height
    
    # Union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def load_yolo_labels(label_path: Union[str, Path]) -> List[List[float]]:
    """
    Load YOLO format labels from a .txt file.
    
    Args:
        label_path: Path to YOLO label file
        
    Returns:
        List of labels, each as [class_id, x_center, y_center, width, height] (normalized)
    """
    labels = []
    label_path = Path(label_path)
    
    if not label_path.exists():
        return labels
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Support space- or comma-separated formats.
            parts = line.split() if ' ' in line else line.split(',')
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                labels.append([class_id, x_center, y_center, width, height])
    
    return labels


def yolo_to_xyxy(box: List[float], img_width: int, img_height: int) -> List[float]:
    """
    Convert YOLO format to absolute pixel coordinates.
    
    Args:
        box: YOLO box [x_center, y_center, width, height] (normalized 0-1)
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        Box in format [x1, y1, x2, y2] (absolute pixels)
    """
    x_center, y_center, width, height = box
    
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    width_abs = width * img_width
    height_abs = height * img_height
    
    x1 = x_center_abs - width_abs / 2
    y1 = y_center_abs - height_abs / 2
    x2 = x_center_abs + width_abs / 2
    y2 = y_center_abs + height_abs / 2
    
    return [x1, y1, x2, y2]


def analyze_tp_fp_fn(
    predictions_json_path: Union[str, Path],
    images_list_path: Union[str, Path, None] = None,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
    single_class: bool = False,
    verbose: bool = True
) -> Dict:
    """
    Analyze True Positives, False Positives, and False Negatives.
    
    Args:
        predictions_json_path: Path to predictions JSON file
        labels_dir: Directory containing ground truth YOLO label files
        images_list_path: Path to text file with list of images (one per line), optional
        iou_threshold: IoU threshold to consider a detection as TP (default: 0.5)
        conf_threshold: Confidence threshold to filter predictions (default: 0.25)
        single_class: If True, ignore class labels and match only by IoU (default: False)
        verbose: Print detailed progress and results (default: True)
        
    Returns:
        Dictionary containing:
            - true_positives: List of TP detections with metadata
            - false_positives: List of FP detections with metadata
            - false_negatives: List of FN detections with metadata
            - n_tp, n_fp, n_fn: Counts
            - precision, recall, f1_score: Performance metrics
            - total_gt_boxes: Total ground truth boxes
            - conf_threshold, iou_threshold: Thresholds used
    """
    predictions_json_path = Path(predictions_json_path)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"ANALYZING TP/FP/FN")
        print(f"  IoU threshold: {iou_threshold}")
        print(f"  Confidence threshold: {conf_threshold}")
        print(f"  Single class mode: {single_class}")
        print(f"  Predictions: {predictions_json_path}")
        if images_list_path:
            print(f"  Images list: {images_list_path}")
        print(f"{'='*80}")
    
    # Load predictions
    with open(predictions_json_path, 'r') as f:
        predictions_list = json.load(f)
    
    # Load images list if provided
    images_paths = []
    if images_list_path:
        images_list_path = Path(images_list_path)
        if images_list_path.exists():
            with open(images_list_path, 'r') as f:
                images_paths = [line.strip() for line in f if line.strip()]
            if verbose:
                print(f"  Loaded {len(images_paths)} image paths from list")
        else:
            if verbose:
                print(f"  Warning: Images list not found at {images_list_path}")
    
    # Initialize counters
    true_positives = []
    false_positives = []
    false_negatives = []
    
    total_gt_boxes = 0
    matched_gt_boxes = 0
    images_processed = 0
    
    # Determine how to map predictions to images
    has_embedded_paths = bool(
        predictions_list
        and isinstance(predictions_list[0], dict)
        and "image" in predictions_list[0]
    )
    if not images_paths and not has_embedded_paths:
        if verbose:
            print("Error: Predictions don't contain image paths. Provide images_list_path or save image paths in JSON.")
        return {
            'true_positives': [],
            'false_positives': [],
            'false_negatives': [],
            'n_tp': 0, 'n_fp': 0, 'n_fn': 0,
            'precision': 0, 'recall': 0, 'f1_score': 0,
            'total_gt_boxes': 0,
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold,
            'images_processed': 0
        }
    
    def _scale_xyxy_if_normalized(box: List[float], w: int, h: int) -> List[float]:
        # If coordinates look normalized, scale to absolute pixels.
        if max(box) <= 1.5:
            return [box[0] * w, box[1] * h, box[2] * w, box[3] * h]
        return box

    # Process each image
    for idx, pred_item in enumerate(predictions_list):
        # Get image path and detections
        if has_embedded_paths and isinstance(pred_item, dict):
            img_path = pred_item.get("image")
            pred_detections = pred_item.get("predictions", pred_item)
        else:
            if idx < len(images_paths):
                img_path = images_paths[idx]
                pred_detections = pred_item
            else:
                if verbose and idx == len(images_paths):
                    print(f"Warning: More predictions ({len(predictions_list)}) than images in list ({len(images_paths)})")
                continue
        if not img_path:
            continue
        
        img_path = Path(img_path)
        
        # Get image dimensions
        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                img_height, img_width = img.shape[:2]
            else:
                if verbose:
                    print(f"Warning: Could not load image {img_path}")
                continue
        else:
            if verbose:
                print(f"Warning: Image not found {img_path}")
            continue
        
        # Get label file path - infer from image path by replacing 'images' with 'labels'
        img_name = img_path.stem
        
        # Try to infer label path from image path (YOLO convention)
        label_path = Path(str(img_path).replace('images', 'labels')).with_suffix('.txt')
        
        # Load ground truth labels (missing label file means no GT objects)
        gt_labels = load_yolo_labels(label_path)
        total_gt_boxes += len(gt_labels)
        
        # Convert GT to absolute coordinates
        gt_boxes_abs = [yolo_to_xyxy(gt[1:5], img_width, img_height) for gt in gt_labels]
        gt_classes = [gt[0] for gt in gt_labels]
        gt_matched = [False] * len(gt_boxes_abs)
        
        # Get predictions for this image (list of detection dicts)
        if isinstance(pred_detections, list):
            pred_boxes = pred_detections
        elif isinstance(pred_detections, dict):
            # Old format compatibility
            pred_boxes = pred_detections.get('boxes', [])
            if not pred_boxes:
                pred_boxes = pred_detections.get('detections', [])
        else:
            pred_boxes = []
        
        # Process each prediction
        for pred in pred_boxes:
            # Handle dict format
            if not isinstance(pred, dict):
                continue
                
            # Get confidence
            conf = pred.get('confidence') or pred.get('conf') or pred.get('score', 0)
            
            # Skip low confidence predictions
            if conf < conf_threshold:
                continue
            
            # Get class (if available)
            pred_class = pred.get('class') or pred.get('class_id')

            # Get box coordinates
            if 'box' in pred:
                box_data = pred['box']
                if isinstance(box_data, dict):
                    pred_box = [box_data['x1'], box_data['y1'], box_data['x2'], box_data['y2']]
                else:
                    continue
            elif 'bbox' in pred:
                bbox = pred['bbox']
                if len(bbox) == 4:
                    pred_box = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                else:
                    continue
            else:
                continue
            pred_box = _scale_xyxy_if_normalized(pred_box, img_width, img_height)
            
            # Find best matching GT box
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes_abs):
                if gt_matched[gt_idx]:
                    continue
                if not single_class and pred_class is not None and pred_class != gt_classes[gt_idx]:
                    continue
                
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Classify detection
            if best_iou >= iou_threshold:
                # True Positive
                gt_matched[best_gt_idx] = True
                true_positives.append({
                    'image': str(img_path),
                    'confidence': conf,
                    'iou': best_iou,
                    'pred_box': pred_box,
                    'gt_box': gt_boxes_abs[best_gt_idx]
                })
            else:
                # False Positive
                false_positives.append({
                    'image': str(img_path),
                    'confidence': conf,
                    'iou': best_iou,
                    'pred_box': pred_box
                })
        
        # Find False Negatives (unmatched GT boxes)
        for gt_idx, matched in enumerate(gt_matched):
            if not matched:
                false_negatives.append({
                    'image': str(img_path),
                    'gt_box': gt_boxes_abs[gt_idx]
                })
        
        matched_gt_boxes += sum(gt_matched)
        images_processed += 1
    
    # Calculate metrics
    n_tp = len(true_positives)
    n_fp = len(false_positives)
    n_fn = len(false_negatives)
    
    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print summary
    if verbose:
        print(f"\n--- Detection Analysis Summary ---")
        print(f"Images processed: {images_processed}")
        print(f"Total ground truth boxes: {total_gt_boxes}")
        print(f"\nDetection Results:")
        print(f"  True Positives (TP):  {n_tp:6d} - Correct detections")
        print(f"  False Positives (FP): {n_fp:6d} - Incorrect detections")
        print(f"  False Negatives (FN): {n_fn:6d} - Missed objects")
        
        print(f"\n--- Performance Metrics ---")
        print(f"  Precision: {precision:.4f} - {precision*100:.2f}% of detections are correct")
        print(f"  Recall:    {recall:.4f} - {recall*100:.2f}% of objects are detected")
        print(f"  F1-Score:  {f1_score:.4f} - Harmonic mean of Precision and Recall")
        
        # Confidence distribution
        if true_positives:
            tp_confs = [tp['confidence'] for tp in true_positives]
            print(f"\n--- True Positives Confidence ---")
            print(f"  Mean: {np.mean(tp_confs):.4f}, Median: {np.median(tp_confs):.4f}")
            print(f"  Range: [{min(tp_confs):.4f}, {max(tp_confs):.4f}]")
        
        if false_positives:
            fp_confs = [fp['confidence'] for fp in false_positives]
            print(f"\n--- False Positives Confidence ---")
            print(f"  Mean: {np.mean(fp_confs):.4f}, Median: {np.median(fp_confs):.4f}")
            print(f"  Range: [{min(fp_confs):.4f}, {max(fp_confs):.4f}]")
        
        # IoU distribution
        if true_positives:
            tp_ious = [tp['iou'] for tp in true_positives]
            print(f"\n--- True Positives IoU ---")
            print(f"  Mean: {np.mean(tp_ious):.4f}, Median: {np.median(tp_ious):.4f}")
            print(f"  Range: [{min(tp_ious):.4f}, {max(tp_ious):.4f}]")
    
    results = {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'n_tp': n_tp,
        'n_fp': n_fp,
        'n_fn': n_fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'total_gt_boxes': total_gt_boxes,
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold,
        'images_processed': images_processed
    }
    
    return results


def visualize_tp_fp_fn_analysis(results: Dict, save_path: Union[str, Path], title_suffix: str = ""):
    """
    Create comprehensive visualizations of TP/FP/FN analysis results.
    
    Args:
        results: Results dictionary from analyze_tp_fp_fn()
        save_path: Path to save the visualization
        title_suffix: Optional suffix to add to the plot title
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. TP/FP/FN Bar Chart
    ax = axes[0, 0]
    categories = ['True\nPositives', 'False\nPositives', 'False\nNegatives']
    counts = [results['n_tp'], results['n_fp'], results['n_fn']]
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title('Detection Categories')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Precision, Recall, F1 Score
    ax = axes[0, 1]
    metrics = ['Precision', 'Recall', 'F1-Score']
    values = [results['precision'], results['recall'], results['f1_score']]
    bars = ax.barh(metrics, values, color=['#3498db', '#9b59b6', '#1abc9c'], 
                   alpha=0.7, edgecolor='black')
    ax.set_xlabel('Score')
    ax.set_xlim([0, 1])
    ax.set_title('Performance Metrics')
    ax.grid(True, alpha=0.3, axis='x')
    
    for bar, value in zip(bars, values):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{value:.3f}', ha='left', va='center', fontweight='bold', fontsize=10)
    
    # 3. Confidence Distribution: TP vs FP
    ax = axes[0, 2]
    if results['true_positives'] and results['false_positives']:
        tp_confs = [tp['confidence'] for tp in results['true_positives']]
        fp_confs = [fp['confidence'] for fp in results['false_positives']]
        
        ax.hist(tp_confs, bins=30, alpha=0.6, label='True Positives', 
                color='green', edgecolor='black')
        ax.hist(fp_confs, bins=30, alpha=0.6, label='False Positives', 
                color='red', edgecolor='black')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Confidence Distribution: TP vs FP')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                transform=ax.transAxes)
        ax.set_title('Confidence Distribution: TP vs FP')
    
    # 4. IoU Distribution for True Positives
    ax = axes[1, 0]
    if results['true_positives']:
        tp_ious = [tp['iou'] for tp in results['true_positives']]
        ax.hist(tp_ious, bins=20, color='green', alpha=0.7, edgecolor='black')
        ax.axvline(results['iou_threshold'], color='red', linestyle='--', linewidth=2,
                   label=f'IoU Threshold: {results["iou_threshold"]}')
        ax.set_xlabel('IoU Score')
        ax.set_ylabel('Frequency')
        ax.set_title('IoU Distribution (True Positives)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No True Positives', ha='center', va='center', 
                transform=ax.transAxes)
        ax.set_title('IoU Distribution (True Positives)')
    
    # 5. Confidence Box Plot: TP vs FP
    ax = axes[1, 1]
    if results['true_positives'] and results['false_positives']:
        tp_confs = [tp['confidence'] for tp in results['true_positives']]
        fp_confs = [fp['confidence'] for fp in results['false_positives']]
        
        bp = ax.boxplot([tp_confs, fp_confs],
                        labels=['True Positives', 'False Positives'],
                        patch_artist=True)
        bp['boxes'][0].set_facecolor('green')
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor('red')
        bp['boxes'][1].set_alpha(0.6)
        
        ax.set_ylabel('Confidence Score')
        ax.set_title('Confidence Box Plot: TP vs FP')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('Confidence Box Plot: TP vs FP')
    
    # 6. Summary Table
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_data = [
        ['Metric', 'Value'],
        ['True Positives', f"{results['n_tp']}"],
        ['False Positives', f"{results['n_fp']}"],
        ['False Negatives', f"{results['n_fn']}"],
        ['Total GT Boxes', f"{results['total_gt_boxes']}"],
        ['', ''],
        ['Precision', f"{results['precision']:.4f}"],
        ['Recall', f"{results['recall']:.4f}"],
        ['F1-Score', f"{results['f1_score']:.4f}"],
        ['', ''],
        ['Conf Threshold', f"{results['conf_threshold']:.2f}"],
        ['IoU Threshold', f"{results['iou_threshold']:.2f}"],
    ]
    
    table = ax.table(cellText=summary_data, cellLoc='left', loc='center',
                     colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style metric rows
    for i in [7, 8, 9]:
        for j in range(2):
            table[(i, j)].set_facecolor('#ecf0f1')
    
    title = f'TP/FP/FN Analysis (Conf≥{results["conf_threshold"]}, IoU≥{results["iou_threshold"]})'
    if title_suffix:
        title += f' - {title_suffix}'
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n✓ Visualization saved to: {save_path}")


def find_optimal_confidence_threshold(
    predictions_json_path: Union[str, Path],
    images_list_path: Union[str, Path],
    iou_threshold: float = 0.5,
    single_class: bool = False,
    num_thresholds: int = 99,
    verbose: bool = True
) -> Dict:
    """
    Find optimal confidence threshold where precision and recall are balanced.
    Sweeps through confidence thresholds to find the first meaningful crossing
    where precision equals recall.
    
    Args:
        predictions_json_path: Path to predictions JSON file
        images_list_path: Path to text file with list of images
        iou_threshold: IoU threshold for matching (default: 0.5)
        single_class: If True, ignore class labels (default: False)
        num_thresholds: Number of thresholds to test (default: 99)
        verbose: Print progress (default: True)
        
    Returns:
        Dictionary containing:
            - best_conf: Optimal confidence threshold
            - best_precision: Precision at optimal threshold
            - best_recall: Recall at optimal threshold
            - selection_note: How the threshold was selected
            - all_thresholds: Array of all tested thresholds
            - all_precisions: Array of precision values
            - all_recalls: Array of recall values
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"FINDING OPTIMAL CONFIDENCE THRESHOLD")
        print(f"  IoU threshold: {iou_threshold}")
        print(f"  Testing {num_thresholds} thresholds from 0.01 to 0.99")
        print(f"{'='*80}")
    
    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    precision_vals = []
    recall_vals = []
    valid_thresholds = []
    
    for conf in thresholds:
        res = analyze_tp_fp_fn(
            predictions_json_path=predictions_json_path,
            images_list_path=images_list_path,
            iou_threshold=iou_threshold,
            conf_threshold=float(conf),
            single_class=single_class,
            verbose=False,
        )
        if res is None:
            continue
        precision = res.get("precision")
        recall = res.get("recall")
        if precision is None or recall is None:
            continue
        precision_vals.append(precision)
        recall_vals.append(recall)
        valid_thresholds.append(float(conf))
    
    if not valid_thresholds:
        if verbose:
            print("✗ No valid precision/recall results.")
        return None
    
    precision_vals = np.array(precision_vals)
    recall_vals = np.array(recall_vals)
    valid_thresholds = np.array(valid_thresholds)
    
    # Find the first meaningful crossing between precision and recall
    # Filter out points where both precision and recall are too low (< 0.1)
    diff = precision_vals - recall_vals
    meaningful_mask = (precision_vals > 0.1) & (recall_vals > 0.1)
    
    # Look for exact matches (where diff ≈ 0) only in meaningful range
    exact_idx = np.where(np.isclose(diff, 0.0, atol=1e-6) & meaningful_mask)[0]
    if exact_idx.size > 0:
        best_idx = int(exact_idx[0])
        best_conf = float(valid_thresholds[best_idx])
        best_precision = float(precision_vals[best_idx])
        best_recall = float(recall_vals[best_idx])
        selection_note = "exact"
    else:
        # Look for sign changes (crossings) only in meaningful range
        sign_changes = np.where((np.sign(diff[:-1]) != np.sign(diff[1:])) & meaningful_mask[:-1])[0]
        if sign_changes.size > 0:
            i = int(sign_changes[0])
            x0, x1 = valid_thresholds[i], valid_thresholds[i + 1]
            y0, y1 = diff[i], diff[i + 1]
            if y1 == y0:
                best_conf = float(x0)
                alpha = 0.0
            else:
                best_conf = float(x0 - y0 * (x1 - x0) / (y1 - y0))
                alpha = (best_conf - x0) / (x1 - x0) if x1 != x0 else 0.0
            best_precision = float(precision_vals[i] + alpha * (precision_vals[i + 1] - precision_vals[i]))
            best_recall = float(recall_vals[i] + alpha * (recall_vals[i + 1] - recall_vals[i]))
            best_idx = i
            selection_note = "first_crossing"
        else:
            # Fallback: find minimum difference in meaningful range
            meaningful_indices = np.where(meaningful_mask)[0]
            if meaningful_indices.size > 0:
                diffs_meaningful = np.abs(diff[meaningful_indices])
                best_idx_in_meaningful = int(np.argmin(diffs_meaningful))
                best_idx = int(meaningful_indices[best_idx_in_meaningful])
                best_conf = float(valid_thresholds[best_idx])
                best_precision = float(precision_vals[best_idx])
                best_recall = float(recall_vals[best_idx])
                selection_note = "closest"
            else:
                # Last resort: use overall minimum
                diffs = np.abs(diff)
                best_idx = int(np.argmin(diffs))
                best_conf = float(valid_thresholds[best_idx])
                best_precision = float(precision_vals[best_idx])
                best_recall = float(recall_vals[best_idx])
                selection_note = "closest"
    
    best_diff = abs(best_precision - best_recall)
    
    if verbose:
        print(f"\n--- Optimal Threshold Found ---")
        print(f"Best confidence threshold: {best_conf:.3f} ({selection_note})")
        print(f"Precision: {best_precision:.4f}")
        print(f"Recall:    {best_recall:.4f}")
        print(f"Abs diff:  {best_diff:.4f}")
    
    return {
        'best_conf': best_conf,
        'best_precision': best_precision,
        'best_recall': best_recall,
        'best_diff': best_diff,
        'selection_note': selection_note,
        'all_thresholds': valid_thresholds,
        'all_precisions': precision_vals,
        'all_recalls': recall_vals
    }


def plot_precision_recall_curve(
    optimal_result: Dict,
    save_path: Union[str, Path] = None,
    title_suffix: str = ""
):
    """
    Plot Precision-Recall curve vs confidence threshold.
    
    Args:
        optimal_result: Result dictionary from find_optimal_confidence_threshold()
        save_path: Path to save the plot (optional)
        title_suffix: Optional suffix for plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    thresholds = optimal_result['all_thresholds']
    precisions = optimal_result['all_precisions']
    recalls = optimal_result['all_recalls']
    best_conf = optimal_result['best_conf']
    best_precision = optimal_result['best_precision']
    best_recall = optimal_result['best_recall']
    
    # Plot 1: Precision/Recall vs Confidence
    ax = axes[0]
    ax.plot(thresholds, precisions, label="Precision", linewidth=2, color='blue')
    ax.plot(thresholds, recalls, label="Recall", linewidth=2, color='purple')
    ax.axvline(best_conf, color="black", linestyle="--", linewidth=2, 
               label=f"Optimal = {best_conf:.3f}")
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision/Recall vs Confidence")
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Precision-Recall Curve
    ax = axes[1]
    ax.plot(recalls, precisions, linewidth=2, color='green')
    ax.scatter([best_recall], [best_precision], color="red", s=150, zorder=5, 
               label=f"Optimal (conf={best_conf:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if title_suffix:
        fig.suptitle(f"Precision-Recall Analysis - {title_suffix}", 
                     fontsize=14, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ PR curve saved to: {save_path}")
    
    plt.show()


def analyze_with_optimal_threshold(
    predictions_json_path: Union[str, Path],
    images_list_path: Union[str, Path],
    iou_threshold: float = 0.5,
    single_class: bool = False,
    save_dir: Union[str, Path] = None,
    verbose: bool = True
) -> Dict:
    """
    Automatically find optimal confidence threshold and run analysis with both
    low threshold (0.01) and optimal threshold.
    
    This function:
    1. Runs initial analysis with conf=0.01 to capture all detections
    2. Sweeps through thresholds to find optimal conf where precision ≈ recall
    3. Runs final analysis with optimal threshold
    4. Returns both results and generates visualizations
    
    Args:
        predictions_json_path: Path to predictions JSON file
        images_list_path: Path to text file with list of images
        iou_threshold: IoU threshold for matching (default: 0.5)
        single_class: If True, ignore class labels (default: False)
        save_dir: Directory to save plots (optional)
        verbose: Print detailed progress (default: True)
        
    Returns:
        Dictionary containing:
            - results_low_threshold: Analysis results with conf=0.01
            - results_optimal: Analysis results with optimal threshold
            - optimal_threshold_info: Information about optimal threshold selection
    """
    if verbose:
        print(f"\n{'#'*80}")
        print(f"# AUTOMATED ANALYSIS WITH OPTIMAL THRESHOLD")
        print(f"{'#'*80}")
    
    # Step 1: Run with low threshold to capture all detections
    if verbose:
        print(f"\n{'='*80}")
        print(f"STEP 1: Analyzing with low confidence threshold (0.01)")
        print(f"{'='*80}")
    
    results_low = analyze_tp_fp_fn(
        predictions_json_path=predictions_json_path,
        images_list_path=images_list_path,
        iou_threshold=iou_threshold,
        conf_threshold=0.01,
        single_class=single_class,
        verbose=verbose
    )
    
    # Step 2: Find optimal threshold
    if verbose:
        print(f"\n{'='*80}")
        print(f"STEP 2: Finding optimal confidence threshold")
        print(f"{'='*80}")
    
    optimal_info = find_optimal_confidence_threshold(
        predictions_json_path=predictions_json_path,
        images_list_path=images_list_path,
        iou_threshold=iou_threshold,
        single_class=single_class,
        verbose=verbose
    )
    
    if optimal_info is None:
        if verbose:
            print("✗ Could not find optimal threshold. Returning only low threshold results.")
        return {
            'results_low_threshold': results_low,
            'results_optimal': None,
            'optimal_threshold_info': None
        }
    
    # Step 3: Run with optimal threshold
    if verbose:
        print(f"\n{'='*80}")
        print(f"STEP 3: Analyzing with optimal confidence threshold ({optimal_info['best_conf']:.3f})")
        print(f"{'='*80}")
    
    results_optimal = analyze_tp_fp_fn(
        predictions_json_path=predictions_json_path,
        images_list_path=images_list_path,
        iou_threshold=iou_threshold,
        conf_threshold=optimal_info['best_conf'],
        single_class=single_class,
        verbose=verbose
    )
    
    # Step 4: Generate visualizations if save_dir provided
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Visualize low threshold results
        visualize_tp_fp_fn_analysis(
            results_low,
            save_dir / "analysis_conf_0.01.png",
            title_suffix="Low Threshold (0.01)"
        )
        
        # Visualize optimal threshold results
        visualize_tp_fp_fn_analysis(
            results_optimal,
            save_dir / f"analysis_conf_{optimal_info['best_conf']:.3f}_optimal.png",
            title_suffix=f"Optimal Threshold ({optimal_info['best_conf']:.3f})"
        )
        
        # Plot PR curve
        plot_precision_recall_curve(
            optimal_info,
            save_dir / "precision_recall_vs_confidence.png",
            title_suffix="Threshold Optimization"
        )
        
        # Create comparison plot
        _plot_low_vs_optimal_comparison(results_low, results_optimal, save_dir)
    
    # Step 5: Print final comparison
    if verbose:
        print(f"\n{'#'*80}")
        print(f"# FINAL COMPARISON")
        print(f"{'#'*80}")
        print(f"\n{'Metric':<20} {'Low (0.01)':<15} {'Optimal ({:.3f})':<15} {'Change':<15}".format(
            optimal_info['best_conf']))
        print("-" * 65)
        print(f"{'Precision':<20} {results_low['precision']:<15.4f} {results_optimal['precision']:<15.4f} "
              f"{results_optimal['precision'] - results_low['precision']:+.4f}")
        print(f"{'Recall':<20} {results_low['recall']:<15.4f} {results_optimal['recall']:<15.4f} "
              f"{results_optimal['recall'] - results_low['recall']:+.4f}")
        print(f"{'F1-Score':<20} {results_low['f1_score']:<15.4f} {results_optimal['f1_score']:<15.4f} "
              f"{results_optimal['f1_score'] - results_low['f1_score']:+.4f}")
        print(f"{'True Positives':<20} {results_low['n_tp']:<15d} {results_optimal['n_tp']:<15d} "
              f"{results_optimal['n_tp'] - results_low['n_tp']:+d}")
        print(f"{'False Positives':<20} {results_low['n_fp']:<15d} {results_optimal['n_fp']:<15d} "
              f"{results_optimal['n_fp'] - results_low['n_fp']:+d}")
        print(f"{'False Negatives':<20} {results_low['n_fn']:<15d} {results_optimal['n_fn']:<15d} "
              f"{results_optimal['n_fn'] - results_low['n_fn']:+d}")
    
    return {
        'results_low_threshold': results_low,
        'results_optimal': results_optimal,
        'optimal_threshold_info': optimal_info
    }


def _plot_low_vs_optimal_comparison(results_low: Dict, results_optimal: Dict, save_dir: Path):
    """Helper function to plot comparison between low and optimal thresholds."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: TP/FP/FN comparison
    ax = axes[0]
    categories = ['TP', 'FP', 'FN']
    low_counts = [results_low['n_tp'], results_low['n_fp'], results_low['n_fn']]
    opt_counts = [results_optimal['n_tp'], results_optimal['n_fp'], results_optimal['n_fn']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, low_counts, width, label='Low (0.01)', alpha=0.7, color='skyblue')
    bars2 = ax.bar(x + width/2, opt_counts, width, label=f'Optimal ({results_optimal["conf_threshold"]:.3f})', 
                   alpha=0.7, color='orange')
    
    ax.set_ylabel('Count')
    ax.set_title('TP/FP/FN: Low vs Optimal Threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add count labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Metrics comparison
    ax = axes[1]
    metrics = ['Precision', 'Recall', 'F1-Score']
    low_metrics = [results_low['precision'], results_low['recall'], results_low['f1_score']]
    opt_metrics = [results_optimal['precision'], results_optimal['recall'], results_optimal['f1_score']]
    
    x = np.arange(len(metrics))
    bars1 = ax.barh(x + width/2, low_metrics, width, label='Low (0.01)', alpha=0.7, color='skyblue')
    bars2 = ax.barh(x - width/2, opt_metrics, width, label=f'Optimal ({results_optimal["conf_threshold"]:.3f})', 
                    alpha=0.7, color='orange')
    
    ax.set_xlabel('Score')
    ax.set_xlim([0, 1])
    ax.set_title('Metrics: Low vs Optimal Threshold')
    ax.set_yticks(x)
    ax.set_yticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            width_val = bar.get_width()
            ax.text(width_val, bar.get_y() + bar.get_height()/2.,
                    f'{width_val:.3f}', ha='left', va='center', fontsize=9)
    
    # Plot 3: Summary table
    ax = axes[2]
    ax.axis('off')
    
    summary_data = [
        ['Metric', 'Low (0.01)', f'Optimal ({results_optimal["conf_threshold"]:.3f})', 'Change'],
        ['Precision', f"{results_low['precision']:.4f}", f"{results_optimal['precision']:.4f}", 
         f"{results_optimal['precision'] - results_low['precision']:+.4f}"],
        ['Recall', f"{results_low['recall']:.4f}", f"{results_optimal['recall']:.4f}",
         f"{results_optimal['recall'] - results_low['recall']:+.4f}"],
        ['F1-Score', f"{results_low['f1_score']:.4f}", f"{results_optimal['f1_score']:.4f}",
         f"{results_optimal['f1_score'] - results_low['f1_score']:+.4f}"],
        ['', '', '', ''],
        ['TP', f"{results_low['n_tp']}", f"{results_optimal['n_tp']}",
         f"{results_optimal['n_tp'] - results_low['n_tp']:+d}"],
        ['FP', f"{results_low['n_fp']}", f"{results_optimal['n_fp']}",
         f"{results_optimal['n_fp'] - results_low['n_fp']:+d}"],
        ['FN', f"{results_low['n_fn']}", f"{results_optimal['n_fn']}",
         f"{results_optimal['n_fn'] - results_low['n_fn']:+d}"],
    ]
    
    table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.25, 0.25, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Comparison Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    save_path = save_dir / 'low_vs_optimal_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Comparison plot saved to: {save_path}")


def compare_thresholds(
    predictions_json_path: Union[str, Path],
    images_list_path: Union[str, Path],
    conf_thresholds: List[float] = [0.25, 0.5, 0.75],
    iou_threshold: float = 0.5,
    single_class: bool = False,
    save_dir: Union[str, Path] = None
) -> Dict[float, Dict]:
    """
    Run analysis across multiple confidence thresholds and compare results.
    
    Args:
        predictions_json_path: Path to predictions JSON
        images_list_path: Path to text file with list of images (one per line)
        conf_thresholds: List of confidence thresholds to test
        iou_threshold: IoU threshold for matching
        single_class: If True, ignore class labels and match only by IoU
        save_dir: Directory to save comparison plots
        
    Returns:
        Dictionary mapping confidence thresholds to their analysis results
    """
    results_by_threshold = {}
    
    print(f"\n{'#'*80}")
    print(f"# COMPARING ACROSS CONFIDENCE THRESHOLDS")
    print(f"{'#'*80}")
    
    for conf_thresh in conf_thresholds:
        print(f"\n{'='*80}")
        print(f"Analyzing with confidence threshold = {conf_thresh}")
        print(f"{'='*80}")
        
        results = analyze_tp_fp_fn(
            predictions_json_path=predictions_json_path,
            images_list_path=images_list_path,
            iou_threshold=iou_threshold,
            conf_threshold=conf_thresh,
            single_class=single_class,
            verbose=True
        )
        
        results_by_threshold[conf_thresh] = results
        
        if save_dir:
            save_path = Path(save_dir) / f'tp_fp_fn_analysis_conf{conf_thresh:.2f}.png'
            visualize_tp_fp_fn_analysis(results, save_path)
    
    # Create comparison visualization
    if save_dir and len(results_by_threshold) > 1:
        _plot_threshold_comparison(results_by_threshold, Path(save_dir))
    
    return results_by_threshold


def _plot_threshold_comparison(results_dict: Dict[float, Dict], save_dir: Path):
    """Helper function to plot comparison across thresholds."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    thresholds = sorted(results_dict.keys())
    
    # Plot 1: TP/FP/FN across thresholds
    ax = axes[0]
    tps = [results_dict[t]['n_tp'] for t in thresholds]
    fps = [results_dict[t]['n_fp'] for t in thresholds]
    fns = [results_dict[t]['n_fn'] for t in thresholds]
    
    x = np.arange(len(thresholds))
    width = 0.25
    
    ax.bar(x - width, tps, width, label='TP', color='green', alpha=0.7)
    ax.bar(x, fps, width, label='FP', color='red', alpha=0.7)
    ax.bar(x + width, fns, width, label='FN', color='orange', alpha=0.7)
    
    ax.set_xlabel('Confidence Threshold')
    ax.set_ylabel('Count')
    ax.set_title('TP/FP/FN by Confidence Threshold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t:.2f}' for t in thresholds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Precision and Recall
    ax = axes[1]
    precisions = [results_dict[t]['precision'] for t in thresholds]
    recalls = [results_dict[t]['recall'] for t in thresholds]
    
    ax.plot(thresholds, precisions, marker='o', linewidth=2, markersize=8,
            label='Precision', color='blue')
    ax.plot(thresholds, recalls, marker='s', linewidth=2, markersize=8,
            label='Recall', color='purple')
    ax.set_xlabel('Confidence Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Precision & Recall by Threshold')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: F1-Score
    ax = axes[2]
    f1_scores = [results_dict[t]['f1_score'] for t in thresholds]
    
    ax.plot(thresholds, f1_scores, marker='D', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Confidence Threshold')
    ax.set_ylabel('F1-Score')
    ax.set_title('F1-Score by Threshold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # Highlight best F1
    best_idx = np.argmax(f1_scores)
    ax.axvline(thresholds[best_idx], color='red', linestyle='--', alpha=0.5,
               label=f'Best: {thresholds[best_idx]:.2f}')
    ax.legend()
    
    plt.tight_layout()
    save_path = save_dir / 'metrics_by_threshold_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n✓ Threshold comparison plot saved to: {save_path}")


# if __name__ == "__main__":
#     # Example usage
#     print("Detection Analysis Utilities")
#     print("Import this module to use the analysis functions:")
#     print("from detection_analysis import analyze_tp_fp_fn, visualize_tp_fp_fn_analysis")
