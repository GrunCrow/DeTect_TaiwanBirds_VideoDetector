# 🎬 Video Annotation Tool for YOLO Training

A modern, efficient web-based video annotation tool for creating YOLO training datasets. Access it from any device on your network via a web browser.

## Features

✨ **Modern Web Interface**
- Clean, responsive design optimized for large datasets
- Access from any browser at `http://YOUR_IP:5000`
- Professional UI with intuitive controls

🎯 **Powerful Annotation**
- Multi-class annotation support (Bat, Bird, DragonFly, Drone, Plane, Other)
- Draw bounding boxes with click & drag
- Zoom and pan for pixel-perfect accuracy
- Automatic coordinate constraints (no invalid boxes)

⚡ **Efficient Workflow**
- Extract N frames from videos (equally distributed)
- View detection preview and video side-by-side
- Keyboard shortcuts for fast navigation (← → arrows)
- Frame timestamps to verify video positions
- Real-time annotation statistics

📦 **YOLO-Ready Output**
- Automatic output directory mirroring with `_annotated` suffix
- Images saved in `images/` folder
- Annotations in YOLO format in `labels/` folder
- Class mapping CSV automatically generated

## Installation

### 1. Install Python Dependencies

```bash
cd f:\DeTect_TaiwanBirds_VideoDetector\label_gui
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python app.py
```

The server will start on `http://localhost:5000`

### 3. Access from Another Machine

From any device on your network:
```
http://YOUR_COMPUTER_IP:5000
```

To find your computer's IP:
- **Windows**: Open PowerShell and run `ipconfig` (look for IPv4 Address)
- **Example**: `http://192.168.1.100:5000`

## Usage

### Step 1: Setup Project
1. Open `http://localhost:5000`
2. Enter the input directory (where your videos and detection images are)
3. Output directory auto-fills with `_annotated` suffix (can be changed)
4. Set number of frames to extract per video (default: 10)
5. Click "Load Videos & Start Annotation"

### Step 2: Annotate Frames
1. View detection preview image on the left
2. Watch the original video in the player
3. Click and drag on the frame canvas to draw bounding boxes
4. Select the correct class for each box
5. Use arrow keys (← →) to navigate between frames
6. Use mouse wheel to zoom in/out for precision
7. Middle mouse button to pan around zoomed image

### Step 3: Save
1. Annotations auto-save when you navigate frames
2. Click "Save All & Exit" to finish and download all annotations

## Directory Structure

```
Input Directory (e.g., G:\2025-04-15_videos)
├── filename_1.jpg          # Detection preview image
├── filename_1.mp4          # Video file
├── filename_2.jpg
├── filename_2.mp4
└── ...

Output Directory (e.g., G:\2025-04-15_videos_annotated)
├── images/
│   ├── filename_1_0.jpg    # Frame 0 extracted
│   ├── filename_1_150.jpg  # Frame at index 150
│   └── ...
├── labels/
│   ├── filename_1_0.txt    # YOLO format annotations
│   ├── filename_1_150.txt
│   └── ...
└── classes.csv             # Class ID mapping
```

## YOLO Label Format

Each `.txt` file contains one annotation per line:
```
class_id x_center y_center width height
```

Where:
- `class_id`: 0=Bat, 1=Bird, 2=DragonFly, 3=Drone, 4=Plane, 5=Other
- `x_center`, `y_center`: Normalized center coordinates (0.0 to 1.0)
- `width`, `height`: Normalized box dimensions (0.0 to 1.0)

### Example
```
1 0.5 0.45 0.3 0.2
2 0.25 0.6 0.15 0.25
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `←` (Left Arrow) | Previous frame |
| `→` (Right Arrow) | Next frame |
| `Del` / `Backspace` | Delete last bounding box |
| `Mouse Wheel` | Zoom in/out |
| Middle Mouse Button | Pan image |

## Classes

| ID | Class |
|----|-------|
| 0 | Bat |
| 1 | Bird |
| 2 | DragonFly |
| 3 | Drone |
| 4 | Plane |
| 5 | Other |

## Tips for Efficiency

✅ **Best Practices**
- Use zoom for accurate bounding boxes
- Check video playback to understand the context
- Use frame timestamps to verify temporal consistency
- Annotate both frame images AND the corresponding frames from the video

⚡ **Performance**
- The tool is optimized for batch processing
- Large datasets are handled efficiently
- Images are served with caching for speed
- Use 10-20 frames per video for good coverage

## Troubleshooting

### Port Already in Use
If port 5000 is already in use, modify `app.py`:
```python
app.run(host='0.0.0.0', port=5001)  # Use port 5001 instead
```

### Video Not Found
Ensure your input directory contains both:
- `.mp4` video files
- `.jpg` detection preview images
- Names must match (e.g., `video_1.mp4` and `video_1.jpg`)

### Images Not Displaying
Check that the input directory path is correct and accessible.
Use absolute paths, not relative paths.

## Performance Notes

- **Frame Extraction**: Frames are extracted on-demand and cached in the output directory
- **Memory**: Images are loaded one at a time, minimizing memory usage
- **Network**: Optimized for local network use (high-speed connections recommended)
- **Concurrency**: Multiple users can work on different videos simultaneously

## File Support

**Video Formats**: MP4, AVI, MOV, FLV
**Image Formats**: JPG, PNG (for detection previews)
**Output**: JPG (extracted frames), TXT (YOLO labels), CSV (class mapping)

## License

Internal Tool - DeTect Taiwan Birds Video Detector

## Support

For issues or feature requests, contact the development team.
