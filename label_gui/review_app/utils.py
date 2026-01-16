"""
Utility functions for the Label Review App
"""
from pathlib import Path
from config import IMAGES_DIR, LABELS_DIR, CLASS_MAPPING, IMAGE_EXTS
from collections import defaultdict
import json
from datetime import datetime


def load_yolo_annotations(label_path):
    """Load YOLO format annotations from a text file.
    Returns list of (class_id, x_center, y_center, width, height) normalized to [0,1]
    """
    bboxes = []
    if isinstance(label_path, str):
        label_path = Path(label_path)
    
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        bboxes.append({
                            'class_id': class_id,
                            'class_name': CLASS_MAPPING.get(class_id, f'Unknown_{class_id}'),
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height
                        })
                    except (ValueError, IndexError):
                        pass
    return bboxes


def save_yolo_annotations(label_path, bboxes):
    """Save bboxes to YOLO format text file."""
    if isinstance(label_path, str):
        label_path = Path(label_path)
    
    with open(label_path, 'w') as f:
        for bbox in bboxes:
            line = f"{bbox['class_id']} {bbox['x_center']} {bbox['y_center']} {bbox['width']} {bbox['height']}\n"
            f.write(line)


def parse_video_and_frame(stem):
    """Extract video name and frame number from filename stem.
    Assumes format: 'videoName_00123'
    """
    parts = stem.rsplit('_', 1)
    if len(parts) == 2:
        video_name = parts[0]
        try:
            frame_number = int(parts[1])
        except ValueError:
            video_name, frame_number = stem, -1
    else:
        video_name, frame_number = stem, -1
    return video_name, frame_number


def load_dataset():
    """Load all images and their annotations.
    Returns dict mapping video_name -> list of frame data
    """
    if not IMAGES_DIR.exists():
        return {}
    
    frames_by_video = defaultdict(list)
    image_files = sorted([p for p in IMAGES_DIR.iterdir() if p.suffix in IMAGE_EXTS])
    
    for img_path in image_files:
        label_path = LABELS_DIR / (img_path.stem + '.txt')
        video_name, frame_number = parse_video_and_frame(img_path.stem)
        bboxes = load_yolo_annotations(label_path)
        
        frame_data = {
            'image_stem': img_path.stem,
            'image_path': str(img_path),
            'label_path': str(label_path),
            'video_name': video_name,
            'frame_number': frame_number,
            'num_targets': len(bboxes),
            'has_annotations': len(bboxes) > 0,
            'bboxes': bboxes
        }
        frames_by_video[video_name].append(frame_data)
    
    # Sort frames by frame number within each video
    for video_name in frames_by_video:
        frames_by_video[video_name].sort(key=lambda x: x['frame_number'])
    
    return dict(frames_by_video)


def get_dataset_stats():
    """Calculate statistics for the entire dataset."""
    dataset = load_dataset()
    
    total_images = sum(len(frames) for frames in dataset.values())
    total_annotated = sum(
        sum(1 for f in frames if f['has_annotations']) 
        for frames in dataset.values()
    )
    total_targets = sum(
        sum(f['num_targets'] for f in frames)
        for frames in dataset.values()
    )
    
    # Class statistics
    class_counts = defaultdict(int)
    images_with_class = defaultdict(int)
    videos_with_class = defaultdict(set)
    
    for video_name, frames in dataset.items():
        for frame in frames:
            for bbox in frame['bboxes']:
                class_name = bbox['class_name']
                class_counts[class_name] += 1
                videos_with_class[class_name].add(video_name)
            
            if frame['has_annotations']:
                for bbox in frame['bboxes']:
                    images_with_class[bbox['class_name']] += 1
    
    # Frames per video
    frames_per_video = {name: len(frames) for name, frames in dataset.items()}
    
    stats = {
        'total_images': total_images,
        'total_annotated': total_annotated,
        'coverage': (total_annotated / total_images * 100) if total_images > 0 else 0,
        'total_targets': total_targets,
        'avg_targets_per_image': total_targets / total_images if total_images > 0 else 0,
        'total_videos': len(dataset),
        'class_counts': dict(class_counts),
        'images_with_class': dict(images_with_class),
        'videos_with_class': {k: len(v) for k, v in videos_with_class.items()},
        'videos_with_class_names': {k: list(v) for k, v in videos_with_class.items()},
        'frames_per_video': frames_per_video,
        'avg_frames_per_video': total_images / len(dataset) if dataset else 0
    }
    
    return stats


def yolo_to_pixel_coords(bbox, img_width, img_height):
    """Convert YOLO normalized coords to pixel coords.
    Returns (x1, y1, x2, y2) in pixels
    """
    x_center = bbox['x_center'] * img_width
    y_center = bbox['y_center'] * img_height
    width = bbox['width'] * img_width
    height = bbox['height'] * img_height
    
    x1 = int(max(0, x_center - width / 2))
    y1 = int(max(0, y_center - height / 2))
    x2 = int(min(img_width, x_center + width / 2))
    y2 = int(min(img_height, y_center + height / 2))
    
    return x1, y1, x2, y2


def log_change(change_record):
    """Log a label change to the changes log.
    change_record should contain: video_path, frame_path, old_class, new_class
    """
    change_record['timestamp'] = datetime.now().isoformat()
    return change_record
