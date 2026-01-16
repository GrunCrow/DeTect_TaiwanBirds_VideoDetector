# TODO: functionality to zoom in/out on images in review app
# TODO: Add possibility to watch full video in review app
# TODO: Add possibility to filter videos by flagged status, relabel history OR CURRENT label classes / number of annotations / number of frames
# TODO: Add possibility to review and change BB creation?
# TODO: Add more information box?

"""
Main Flask application for Label Review App
"""
from flask import Flask, render_template, jsonify, request, send_file
from pathlib import Path
from config import HOST, PORT, CLASS_MAPPING, IMAGES_DIR
from utils import load_dataset, get_dataset_stats, load_yolo_annotations, save_yolo_annotations, log_change, yolo_to_pixel_coords
from PIL import Image, ImageDraw
import csv
from datetime import datetime
import io
import json

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Session storage for changes
session_changes = []

FLAGGED_VIDEOS_CSV = Path('tracking/bb_videos_to_review.csv')


def load_flagged_videos():
    """Load list of flagged videos from CSV."""
    flagged = {}
    if not FLAGGED_VIDEOS_CSV.exists():
        return flagged
    
    try:
        with open(FLAGGED_VIDEOS_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_name = row.get('video_name', '')
                if video_name:
                    flagged[video_name] = {
                        'timestamp': row.get('timestamp', ''),
                        'annotator': row.get('annotator', ''),
                        'reason': row.get('reason', '')
                    }
    except Exception as e:
        print(f"[ERROR] Failed to load flagged videos: {str(e)}")
    
    return flagged


def save_flagged_video(video_name, annotator, reason=''):
    """Add a video to the flagged list."""
    # Check if already flagged
    flagged = load_flagged_videos()
    if video_name in flagged:
        return  # Already flagged
    
    # Create file with header if it doesn't exist
    file_exists = FLAGGED_VIDEOS_CSV.exists()
    
    with open(FLAGGED_VIDEOS_CSV, 'a', newline='', encoding='utf-8') as f:
        fieldnames = ['video_name', 'timestamp', 'annotator', 'reason']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'video_name': video_name,
            'timestamp': datetime.now().isoformat(),
            'annotator': annotator,
            'reason': reason
        })


def remove_flagged_video(video_name):
    """Remove a video from the flagged list."""
    if not FLAGGED_VIDEOS_CSV.exists():
        return
    
    # Read all entries except the one to remove
    entries = []
    with open(FLAGGED_VIDEOS_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        entries = [row for row in reader if row.get('video_name') != video_name]
    
    # Rewrite the file
    with open(FLAGGED_VIDEOS_CSV, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['video_name', 'timestamp', 'annotator', 'reason']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(entries)


def load_relabel_history():
    """Load relabel history from tracking files."""
    history = {}
    tracking_dir = Path('tracking/relabel_tracking')
    
    print(f"[DEBUG] Looking for relabel history in: {tracking_dir}")
    print(f"[DEBUG] Tracking dir exists: {tracking_dir.exists()}")
    
    if not tracking_dir.exists():
        print(f"[DEBUG] Tracking directory does not exist")
        return history
    
    # Read all CSV files in the tracking directory
    csv_files = list(tracking_dir.glob('relabel_log_*.csv'))
    print(f"[DEBUG] Found {len(csv_files)} CSV files")
    
    for csv_file in csv_files:
        print(f"[DEBUG] Reading file: {csv_file}")
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    video_name = row.get('video', '')
                    if video_name:
                        if video_name not in history:
                            history[video_name] = {
                                'reviewers': set(),
                                'changes': []
                            }
                        
                        history[video_name]['reviewers'].add(row.get('annotator', ''))
                        history[video_name]['changes'].append({
                            'annotator': row.get('annotator', ''),
                            'old_class': row.get('old_class', ''),
                            'new_class': row.get('new_class', ''),
                            'timestamp': row.get('timestamp', '')
                        })
                        print(f"[DEBUG] Added change for {video_name}: {row.get('old_class', '')} -> {row.get('new_class', '')}")
        except Exception as e:
            print(f"[ERROR] Failed to read {csv_file}: {str(e)}")
    
    # Convert sets to lists for JSON serialization
    for video_name in history:
        history[video_name]['reviewers'] = list(history[video_name]['reviewers'])
    
    print(f"[DEBUG] Total videos with history: {len(history)}")
    for video_name in history:
        print(f"[DEBUG] {video_name}: {len(history[video_name]['changes'])} changes by {history[video_name]['reviewers']}")
    
    return history


@app.route('/')
def index():
    """Render the dashboard."""
    return render_template('dashboard.html')


@app.route('/api/stats')
def api_stats():
    """Get dataset statistics."""
    try:
        print(f"Loading dataset from: {IMAGES_DIR}")
        stats = get_dataset_stats()
        print(f"Stats loaded successfully: {stats.get('total_images', 0)} images")
        return jsonify(stats)
    except Exception as e:
        import traceback
        error_msg = f"Error loading stats: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/videos')
def api_videos():
    """Get list of videos with frame counts."""
    try:
        dataset = load_dataset()
        relabel_history = load_relabel_history()
        flagged_videos = load_flagged_videos()
        
        videos = []
        for video_name, frames in sorted(dataset.items()):
            annotated_count = sum(1 for f in frames if f['has_annotations'])
            video_info = {
                'name': video_name,
                'total_frames': len(frames),
                'annotated_frames': annotated_count,
                'background_frames': len(frames) - annotated_count
            }
            
            # Add relabel history if available
            if video_name in relabel_history:
                video_info['relabel_history'] = relabel_history[video_name]
            
            # Add flagged status if video is flagged
            if video_name in flagged_videos:
                video_info['flagged'] = True
                video_info['flag_info'] = flagged_videos[video_name]
            else:
                video_info['flagged'] = False
            
            videos.append(video_info)
        return jsonify(videos)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/video/<video_name>/flag', methods=['POST'])
def api_toggle_flag(video_name):
    """Toggle flag status for a video."""
    try:
        data = request.get_json()
        flagged = data.get('flagged', False)
        annotator = data.get('annotator', 'Unknown')
        reason = data.get('reason', '')
        
        if flagged:
            save_flagged_video(video_name, annotator, reason)
            print(f"[INFO] Video '{video_name}' flagged by {annotator}")
        else:
            remove_flagged_video(video_name)
            print(f"[INFO] Video '{video_name}' unflagged")
        
        return jsonify({'success': True, 'flagged': flagged})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/video/<video_name>')
def api_video_frames(video_name):
    """Get all frames for a video."""
    try:
        dataset = load_dataset()
        if video_name not in dataset:
            return jsonify({'error': 'Video not found'}), 404
        
        frames = dataset[video_name]
        frames_data = []
        
        for frame in frames:
            frames_data.append({
                'stem': frame['image_stem'],
                'image_path': frame['image_path'],
                'label_path': frame['label_path'],
                'frame_number': frame['frame_number'],
                'num_targets': frame['num_targets'],
                'has_annotations': frame['has_annotations'],
                'bboxes': frame['bboxes']
            })
        
        return jsonify(frames_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/frame/image/<path:image_path>')
def api_frame_image(image_path):
    """Serve a frame image with optional bboxes drawn."""
    try:
        draw_bboxes = request.args.get('draw_bboxes', 'true').lower() == 'true'
        selected_bbox_index = request.args.get('selected_bbox_index', type=int, default=None)
        
        # The image_path is sent as a full path from the frontend
        full_path = Path(image_path)
        print(f"[DEBUG] Requested image path: {image_path}")
        print(f"[DEBUG] Full path: {full_path}")
        print(f"[DEBUG] Path exists: {full_path.exists()}")
        print(f"[DEBUG] Selected bbox index: {selected_bbox_index}")
        
        if not full_path.exists():
            print(f"[DEBUG] Image not found at: {full_path}")
            return jsonify({'error': f'Image not found at {full_path}'}), 404
        
        # Open image
        img = Image.open(full_path).convert('RGB')
        print(f"[DEBUG] Image opened: {img.size}")
        
        if draw_bboxes:
            # Draw bboxes if requested
            label_path = full_path.parent.parent / 'labels' / (full_path.stem + '.txt')
            print(f"[DEBUG] Looking for labels at: {label_path}")
            if label_path.exists():
                bboxes = load_yolo_annotations(label_path)
                print(f"[DEBUG] Found {len(bboxes)} bboxes")
                img = draw_bboxes_on_image(img, bboxes, selected_bbox_index)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        print(f"[DEBUG] Image served successfully")
        return send_file(img_byte_arr, mimetype='image/png')
    except Exception as e:
        import traceback
        print(f"[ERROR] Failed to serve image: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


def draw_bboxes_on_image(img, bboxes, selected_bbox_index=None):
    """Draw bounding boxes on image using PIL."""
    
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    color_map = {
        'Bat': (255, 0, 0),
        'Bird': (0, 255, 0),
        'Insect': (0, 0, 255),
        'Drone': (255, 255, 0),
        'Plane': (255, 0, 255),
        'Other': (0, 255, 255),
        'Unknown': (128, 128, 128)
    }
    
    for idx, bbox in enumerate(bboxes):
        class_name = bbox['class_name']
        color = color_map.get(class_name, (0, 255, 0))
        
        # Highlight selected bbox
        if idx == selected_bbox_index:
            color = (255, 255, 255)  # White for selected
            line_width = 4
        else:
            line_width = 2
        
        x1, y1, x2, y2 = yolo_to_pixel_coords(bbox, width, height)
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        
        # Draw label
        label = f"{class_name}"
        text_bbox = draw.textbbox((x1, max(0, y1-15)), label)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Use the original color for label background, or white if selected
        label_bg_color = (255, 255, 255) if idx == selected_bbox_index else color
        draw.rectangle([x1, max(0, y1-text_height-4), x1+text_width+4, y1], fill=label_bg_color)
        draw.text((x1+2, max(2, y1-text_height-2)), label, fill=(255, 255, 255))
    
    return img


@app.route('/api/frame/update-class', methods=['POST'])
def api_update_class():
    """Update the class of a specific bounding box."""
    try:
        data = request.json
        frame_image_path = data.get('frame_image_path')
        bbox_index = data.get('bbox_index')
        new_class_id = data.get('new_class_id')
        old_class_name = data.get('old_class_name')
        new_class_name = CLASS_MAPPING.get(new_class_id, f'Unknown_{new_class_id}')
        
        label_path = Path(frame_image_path).parent.parent / 'labels' / (Path(frame_image_path).stem + '.txt')
        
        if not label_path.exists():
            return jsonify({'error': 'Label file not found'}), 404
        
        # Load current annotations
        bboxes = load_yolo_annotations(label_path)
        
        if bbox_index >= len(bboxes):
            return jsonify({'error': 'Bbox index out of range'}), 400
        
        # Update class
        bboxes[bbox_index]['class_id'] = new_class_id
        bboxes[bbox_index]['class_name'] = new_class_name
        
        # Save updated annotations
        save_yolo_annotations(label_path, bboxes)
        
        # Automatically save to tracking file (concurrent-safe append)
        try:
            tracking_dir = Path('relabel_tracking')
            tracking_dir.mkdir(exist_ok=True)
            
            tracking_file = tracking_dir / f'relabel_log_{datetime.now().strftime("%Y%m%d")}.csv'
            
            # Get annotator from request headers or use default
            annotator = request.headers.get('X-Annotator', 'Unknown')
            
            # Extract video name from the frame filename stem
            # Filename format: video_name_frameNumber.jpg
            # Video name is everything before the last underscore
            frame_stem = Path(frame_image_path).stem
            video_name = frame_stem.rsplit('_', 1)[0] if '_' in frame_stem else frame_stem
            
            print(f"[DEBUG] Extracted video_name: {video_name} from frame_stem: {frame_stem}")
            
            # Prepare row data
            row_data = {
                'timestamp': datetime.now().isoformat(),
                'annotator': annotator,
                'video': video_name,
                'frame': Path(frame_image_path).stem,
                'old_class': old_class_name,
                'new_class': new_class_name,
                'image_path': str(frame_image_path)
            }
            
            # Append to CSV in a concurrent-safe manner
            file_exists = tracking_file.exists()
            with open(tracking_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=['timestamp', 'annotator', 'video', 'frame', 'old_class', 'new_class', 'image_path']
                )
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row_data)
            
            print(f"[AUTO-TRACKING] Saved change: {video_name} - {old_class_name} → {new_class_name} by {annotator}")
        except Exception as tracking_error:
            print(f"[WARNING] Failed to auto-track change: {str(tracking_error)}")
            # Don't fail the entire operation if tracking fails
        
        # Log the change
        change = {
            'video_path': str(frame_image_path),
            'frame_path': str(label_path),
            'old_class': old_class_name,
            'new_class': new_class_name,
            'bbox_index': bbox_index,
            'timestamp': datetime.now().isoformat()
        }
        session_changes.append(change)
        
        return jsonify({
            'success': True,
            'message': f'Updated bbox {bbox_index} from {old_class_name} to {new_class_name}',
            'change_id': len(session_changes) - 1
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/changes')
def api_get_changes():
    """Get all recorded changes in this session."""
    return jsonify({
        'total_changes': len(session_changes),
        'changes': session_changes
    })


@app.route('/api/changes/export', methods=['POST'])
def api_export_changes():
    """Export changes as CSV."""
    try:
        data = request.json
        changes = data.get('changes', [])
        annotator = data.get('annotator', 'Unknown')
        
        if not changes:
            return jsonify({'error': 'No changes to export'}), 400
        
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=['video', 'frame', 'old_class', 'new_class', 'annotator', 'timestamp']
        )
        writer.writeheader()
        
        timestamp = datetime.now().isoformat()
        for change in changes:
            writer.writerow({
                'video': change.get('video', ''),
                'frame': change.get('frame', ''),
                'old_class': change.get('old_class', ''),
                'new_class': change.get('new_class', ''),
                'annotator': annotator,
                'timestamp': timestamp
            })
        
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'label_changes_{annotator}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/changes/clear', methods=['POST'])
def api_clear_changes():
    """Clear all recorded changes."""
    global session_changes
    session_changes = []
    return jsonify({'success': True, 'message': 'Changes cleared'})


@app.route('/api/relabel/save', methods=['POST'])
def api_relabel_save():
    """Save relabeling changes to tracking file."""
    try:
        data = request.json
        video_name = data.get('video')
        changes = data.get('changes', [])
        annotator = data.get('annotator', 'Unknown')
        
        if not changes:
            return jsonify({'error': 'No changes to save'}), 400
        
        # Create tracking directory if it doesn't exist
        tracking_dir = Path('relabel_tracking')
        tracking_dir.mkdir(exist_ok=True)
        
        # Save to a CSV file
        tracking_file = tracking_dir / f'relabel_log_{datetime.now().strftime("%Y%m%d")}.csv'
        
        # Prepare data for CSV
        rows = []
        timestamp = datetime.now().isoformat()
        for change in changes:
            rows.append({
                'timestamp': timestamp,
                'annotator': annotator,
                'video': video_name,
                'frame': change.get('frame', ''),
                'old_class': change.get('oldClass', ''),
                'new_class': change.get('newClass', ''),
                'image_path': change.get('image_path', '')
            })
        
        # Append to tracking file
        file_exists = tracking_file.exists()
        with open(tracking_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['timestamp', 'annotator', 'video', 'frame', 'old_class', 'new_class', 'image_path']
            )
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)
        
        print(f"[RELABEL] Saved {len(changes)} changes for {video_name} by {annotator} to {tracking_file}")
        
        return jsonify({
            'success': True,
            'message': f'Saved {len(changes)} changes to {tracking_file}',
            'file': str(tracking_file)
        })
    except Exception as e:
        import traceback
        print(f"[ERROR] {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/review')
def review():
    """Render the review page."""
    return render_template('review.html')


@app.route('/api/classes')
def api_classes():
    """Get list of available classes."""
    return jsonify(CLASS_MAPPING)


if __name__ == '__main__':
    print(f"ðŸš€ Starting Label Review App")
    print(f"ðŸ“Š Dashboard: http://{HOST}:{PORT}/")
    print(f"ðŸŽ¬ Video Review: http://{HOST}:{PORT}/review")
    app.run(host=HOST, port=PORT, debug=True, use_reloader=False)
