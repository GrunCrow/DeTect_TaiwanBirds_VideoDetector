import os
import csv
import cv2
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import threading
from datetime import datetime

################################################################################
# Constants
################################################################################

CLASSES = ["Bat", "Bird", "DragonFly", "Drone", "Plane", "Other"]
CLASS_MAPPING = {name: idx for idx, name in enumerate(CLASSES)}
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'flv'}

################################################################################
# Initialize Flask App
################################################################################

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1GB max file size
app.config['JSON_SORT_KEYS'] = False

# Global state
annotation_state = {
    'input_dir': None,
    'output_dir': None,
    'num_frames': 10,
    'video_files': [],
    'current_video_idx': 0,
    'extracted_frames': {},  # {video_basename: [(frame_idx, frame_path), ...]}
    'annotations': {},  # {frame_path: [(class_id, x, y, w, h), ...]}
}

################################################################################
# Utility Functions
################################################################################

def extract_basename(path):
    """Extract filename without extension"""
    return os.path.splitext(os.path.basename(path))[0]

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)

def get_mirror_output_path(input_dir):
    """Generate mirror output directory with _annotated suffix"""
    parent = os.path.dirname(input_dir)
    basename = os.path.basename(input_dir)
    return os.path.join(parent, basename + "_annotated")

def extract_frames_from_video(video_path, num_frames=10, output_images_dir=None):
    """Extract N equally distributed frames from video"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return []
    
    # Calculate frame indices to extract
    if num_frames >= total_frames:
        indices = list(range(total_frames))
    else:
        step = (total_frames - 1) / (num_frames - 1) if num_frames > 1 else 0
        indices = [int(i * step) for i in range(num_frames)]
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_name = f"{extract_basename(video_path)}_{idx}.jpg"
            if output_images_dir:
                frame_path = os.path.join(output_images_dir, frame_name)
                cv2.imwrite(frame_path, frame)
            frames.append((idx, frame_name))
    
    cap.release()
    return frames

def get_frame_timestamp(video_path, frame_idx):
    """Get timestamp of a specific frame in the video"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    if fps > 0:
        timestamp_sec = frame_idx / fps
        minutes = int(timestamp_sec // 60)
        seconds = timestamp_sec % 60
        return f"{minutes:02d}:{seconds:05.2f}"
    return "00:00.00"

def save_class_mapping(output_path):
    """Save class mapping to CSV file"""
    csv_path = os.path.join(output_path, "classes.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name"])
        for class_name, class_id in CLASS_MAPPING.items():
            writer.writerow([class_id, class_name])

################################################################################
# Flask Routes - API
################################################################################

@app.route('/api/init', methods=['POST'])
def init_annotation():
    """Initialize annotation with input/output directories"""
    data = request.json
    input_dir = data.get('input_dir')
    output_dir = data.get('output_dir')
    num_frames = data.get('num_frames', 10)
    
    if not input_dir or not os.path.exists(input_dir):
        return jsonify({'error': 'Invalid input directory'}), 400
    
    # Find all video files
    all_files = os.listdir(input_dir)
    video_files = [f for f in all_files if f.lower().endswith(tuple(ALLOWED_EXTENSIONS))]
    
    if not video_files:
        return jsonify({'error': 'No video files found'}), 400
    
    video_files = sorted([extract_basename(f) for f in video_files])
    
    # Create output structure
    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, "images"))
    ensure_dir(os.path.join(output_dir, "labels"))
    
    # Save class mapping
    save_class_mapping(output_dir)
    
    # Update state
    annotation_state['input_dir'] = input_dir
    annotation_state['output_dir'] = output_dir
    annotation_state['num_frames'] = num_frames
    annotation_state['video_files'] = video_files
    annotation_state['current_video_idx'] = 0
    annotation_state['extracted_frames'] = {}
    annotation_state['annotations'] = {}
    
    return jsonify({
        'success': True,
        'video_count': len(video_files),
        'message': f'Loaded {len(video_files)} video(s)'
    })

@app.route('/api/video/<int:video_idx>/frames', methods=['GET'])
def get_video_frames(video_idx):
    """Get frames for a specific video"""
    if video_idx >= len(annotation_state['video_files']):
        return jsonify({'error': 'Invalid video index'}), 400
    
    basename = annotation_state['video_files'][video_idx]
    
    if basename in annotation_state['extracted_frames']:
        frames = annotation_state['extracted_frames'][basename]
    else:
        input_dir = annotation_state['input_dir']
        output_dir = annotation_state['output_dir']
        images_dir = os.path.join(output_dir, "images")
        
        video_path = os.path.join(input_dir, basename + ".mp4")
        if not os.path.exists(video_path):
            # Try other extensions
            for ext in ALLOWED_EXTENSIONS:
                video_path = os.path.join(input_dir, basename + "." + ext)
                if os.path.exists(video_path):
                    break
        
        if not os.path.exists(video_path):
            return jsonify({'error': f'Video not found: {basename}'}), 400
        
        frames = extract_frames_from_video(video_path, annotation_state['num_frames'], images_dir)
        annotation_state['extracted_frames'][basename] = frames
    
    return jsonify({
        'success': True,
        'video_basename': basename,
        'video_idx': video_idx,
        'total_videos': len(annotation_state['video_files']),
        'frames': frames,
        'frame_count': len(frames)
    })

@app.route('/api/frame/<path:frame_name>/image', methods=['GET'])
def get_frame_image(frame_name):
    """Get frame image"""
    output_dir = annotation_state['output_dir']
    img_path = os.path.join(output_dir, "images", frame_name)
    
    if not os.path.exists(img_path):
        return jsonify({'error': 'Frame not found'}), 404
    
    return send_file(img_path, mimetype='image/jpeg')

@app.route('/api/frame/<path:frame_name>/annotations', methods=['GET'])
def get_frame_annotations(frame_name):
    """Get existing annotations for a frame"""
    output_dir = annotation_state['output_dir']
    label_path = os.path.join(output_dir, "labels", frame_name.replace('.jpg', '.txt'))
    
    annotations = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    annotations.append({
                        'class_id': class_id,
                        'class_name': CLASSES[class_id],
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    })
    
    return jsonify({
        'success': True,
        'frame_name': frame_name,
        'annotations': annotations
    })

@app.route('/api/frame/<path:frame_name>/save-annotations', methods=['POST'])
def save_frame_annotations(frame_name):
    """Save annotations for a frame"""
    data = request.json
    annotations = data.get('annotations', [])
    
    output_dir = annotation_state['output_dir']
    label_path = os.path.join(output_dir, "labels", frame_name.replace('.jpg', '.txt'))
    
    if annotations:
        with open(label_path, 'w') as f:
            for anno in annotations:
                class_id = anno['class_id']
                x_center = max(0.0, min(1.0, anno['x_center']))
                y_center = max(0.0, min(1.0, anno['y_center']))
                width = max(0.0, min(1.0, anno['width']))
                height = max(0.0, min(1.0, anno['height']))
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    else:
        # Delete file if no annotations
        if os.path.exists(label_path):
            os.remove(label_path)
    
    return jsonify({'success': True, 'message': 'Annotations saved'})

@app.route('/api/detection-image/<path:video_basename>', methods=['GET'])
def get_detection_image(video_basename):
    """Get detection preview image for a video"""
    input_dir = annotation_state['input_dir']
    img_path = os.path.join(input_dir, video_basename + ".jpg")
    
    if not os.path.exists(img_path):
        return jsonify({'error': 'Image not found'}), 404
    
    return send_file(img_path, mimetype='image/jpeg')

@app.route('/api/video/<path:video_basename>/timestamp/<int:frame_idx>', methods=['GET'])
def get_video_timestamp(video_basename, frame_idx):
    """Get timestamp for a frame"""
    input_dir = annotation_state['input_dir']
    
    video_path = os.path.join(input_dir, video_basename + ".mp4")
    if not os.path.exists(video_path):
        for ext in ALLOWED_EXTENSIONS:
            video_path = os.path.join(input_dir, video_basename + "." + ext)
            if os.path.exists(video_path):
                break
    
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video not found'}), 404
    
    timestamp = get_frame_timestamp(video_path, frame_idx)
    return jsonify({'success': True, 'timestamp': timestamp})

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get list of classes"""
    return jsonify({
        'success': True,
        'classes': CLASSES,
        'mapping': CLASS_MAPPING
    })

################################################################################
# Flask Routes - Pages
################################################################################

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/annotate')
def annotate():
    """Annotation page"""
    return render_template('annotate.html')

################################################################################
# Main
################################################################################

if __name__ == '__main__':
    print("Starting Video Annotation Web Server...")
    print("Open your browser and go to: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
