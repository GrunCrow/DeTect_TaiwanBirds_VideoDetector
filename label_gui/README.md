# Video Annotation Tool for YOLO Training

## 📋 Overview

A professional desktop application for annotating videos to create training datasets for YOLO object detection models. The tool extracts frames from videos, allows you to draw bounding boxes around objects, and exports annotations in YOLO format.

## 🎯 Supported Classes

The tool supports 6 object classes:
- **Bat** (ID: 0)
- **Bird** (ID: 1)
- **DragonFly** (ID: 2)
- **Drone** (ID: 3)
- **Plane** (ID: 4)
- **Other** (ID: 5)

## 🚀 Getting Started

### Initial Setup

1. **Select Input Directory**: Choose the folder containing your video files (.mp4) and detection images (.jpg)
2. **Select Output Directory**: Choose where to save annotated frames and labels
3. **Set Number of Frames**: Specify how many frames to extract from each video (default: 10)
4. **Load Videos**: Click "🚀 Load Videos" to start the annotation process

### Progress Tracking

The application automatically saves your progress:
- A `.progress.json` file tracks the last reviewed video
- When you reload videos, the app resumes from where you left off
- Progress bar shows overall completion percentage

## 🖱️ Mouse Controls

### Frame Annotation Canvas

| Action | Mouse Control |
|--------|---------------|
| **Draw Bounding Box** | Left-click and drag |
| **Delete Bounding Box** | Right-click inside the bounding box |
| **Pan Image** | Middle-click and drag |
| **Zoom In/Out** | Mouse wheel scroll up/down |

### Detection Preview

| Action | Mouse Control |
|--------|---------------|
| **Zoom In/Out** | Mouse wheel scroll up/down |
| **Reset Zoom** | Double-click |

## ⌨️ Keyboard Shortcuts

### Frame Navigation

| Key | Action |
|-----|--------|
| **Left Arrow** (`←`) | Previous frame |
| **Right Arrow** (`→`) | Next frame |
| **Delete** or **Backspace** | Delete last bounding box |
| **Enter** | Save current video and move to next video |

## 🎨 Annotation Workflow

### Step-by-Step Process

1. **Load Videos**: Use the "Load Videos" button to import your video files
2. **Select Class**: Choose the object class from the dropdown menu (Bat, Bird, etc.)
3. **Draw Bounding Box**: 
   - Click and drag on the frame to draw a rectangle around the object
   - The box will be color-coded by class
   - Class name label appears at the top-left of each box
4. **Navigate Frames**: Use arrow keys or buttons to move between frames
5. **Delete if Needed**: 
   - Right-click inside a box to delete it
   - Press Delete/Backspace to remove the last drawn box
6. **Move to Next Video**: Press Enter or click "Next Video" when done with all frames
7. **Review & Adjust**: Use "Previous Video" button to go back if needed

### Zoom and Pan

- **Zoom**: Use mouse wheel to zoom in/out for precise annotation
- **Pan**: Middle-click and drag to move around when zoomed in
- **Reset**: Click "Reset Zoom" button to fit the image back to canvas
- **Zoom Constraints**: Minimum zoom is auto-calculated to fit the entire image

## 📁 Output Structure

The output directory contains:

```
output_directory/
├── .progress.json          # Progress tracking file
├── classes.csv             # Class mapping (class_id, class_name)
├── images/                 # Extracted frame images
│   ├── video1_0.jpg
│   ├── video1_100.jpg
│   └── ...
└── labels/                 # YOLO format annotations
    ├── video1_0.txt
    ├── video1_100.txt
    └── ...
```

### YOLO Annotation Format

Each `.txt` file contains one line per bounding box:
```
class_id x_center y_center width height
```

All coordinates are normalized to [0, 1] range:
- `x_center`: Center X coordinate (0.0 to 1.0)
- `y_center`: Center Y coordinate (0.0 to 1.0)
- `width`: Box width relative to image width
- `height`: Box height relative to image height

Example:
```
1 0.502341 0.487234 0.123456 0.234567
```

## 🎮 UI Components

### Top Control Panel

- **Input Directory**: Path to video files
- **Output Directory**: Path to save annotations
- **Number of Frames**: Frames to extract per video (1-100)
- **Load Videos**: Initialize the annotation session

### Left Panel

#### Detection Preview Card
- Shows the detection image (.jpg) for the current video
- Supports zoom (mouse wheel) and reset (double-click)
- Zoom level indicator appears when zoomed

#### Video Player Card
- Displays the current video
- **Play** (▶): Start video playback
- **Pause** (⏸): Pause video playback
- **Frame Time** (⏱️): Show timestamp of current frame

#### Class Selection Card
- Dropdown menu to select annotation class
- Current class is displayed and used for new bounding boxes

### Right Panel

#### Info & Progress
- **Video/Frame Counter**: Shows current position (e.g., "Video 5/20: filename | Frame 3/10")
- **Progress Bar**: Visual indicator of overall completion
- **Status Box**: Non-blocking messages for workflow feedback

#### Frame Canvas
- Main annotation area
- Displays current frame with existing annotations
- Click and drag to create new bounding boxes
- Zoom level indicator (bottom-left when zoomed)

#### Control Buttons

**Row 1 - Frame Navigation:**
- **◀ Prev (←)**: Previous frame
- **Next (→) ▶**: Next frame
- **🗑️ Delete (Del)**: Delete last bounding box

**Row 2 - Video & Zoom:**
- **🔍 Reset Zoom**: Fit and center image
- **◀◀ Previous Video**: Go to previous video
- **▶▶ Next Video (Enter)**: Save and move to next video
- **💾 Save All & Exit**: Save all annotations and close

## 🎨 Visual Features

### Color-Coded Bounding Boxes

Each class has a distinct color:
- **Red**: Bat (0)
- **Blue**: Bird (1)
- **Green**: DragonFly (2)
- **Yellow**: Drone (3)
- **Purple**: Plane (4)
- **Orange**: Other (5)

### Status Messages

The status box shows real-time feedback:
- ✅ **Green**: Success messages (videos loaded, saved)
- 📌 **Blue**: Information (last frame reached, navigation)
- ⚠️ **Orange**: Warnings (already at first video)

## 💡 Tips & Best Practices

1. **Start with Zoom**: When objects are small, zoom in before drawing bounding boxes
2. **Use Keyboard Shortcuts**: Arrow keys and Enter make annotation much faster
3. **Save Frequently**: Press Enter to save each video as you complete it
4. **Review with Previous Video**: Use the Previous Video button to check your work
5. **Check Progress Bar**: Keep track of how many videos you've completed
6. **Load Existing Annotations**: The tool automatically loads and displays existing annotations when you reopen videos
7. **Modify Existing Boxes**: Right-click to delete incorrect boxes and redraw them
8. **Frame Selection**: Adjust the number of frames per video based on video content (more frames = more training data but slower annotation)

## 📊 Workflow Efficiency

### Fast Annotation Mode

1. Select class from dropdown
2. Draw all bounding boxes for current frame
3. Press **Right Arrow** to move to next frame
4. Repeat until all frames are annotated
5. Press **Enter** to save and move to next video
6. No modal dialogs - all feedback in status box!

### Editing Mode

1. Use **Left/Right Arrows** to navigate frames
2. **Right-click** on incorrect bounding boxes to delete them
3. Redraw corrected bounding boxes
4. Changes are auto-saved when moving to next frame or video

## 🔄 Progress Recovery

If you close the application:
- Progress is automatically saved in `.progress.json`
- Next time you load the same output directory, you'll resume from the last video
- All previously saved annotations are preserved and loaded

## 🖥️ System Requirements

- **Python 3.7+**
- **Libraries**: tkinter, OpenCV (cv2), Pillow (PIL), threading
- **Display**: Recommended minimum 1920x1080 resolution
- **Input**: Mouse (with scroll wheel recommended), Keyboard

## 📝 Notes

- Only `.mp4` video files are supported
- Detection images must be `.jpg` format with the same basename as videos
- Minimum bounding box size: 5 pixels (very small boxes are ignored)
- Zoom range: 0.5x to 10.0x for frame canvas, 0.5x to 3.0x for detection preview
- Frame extraction distributes frames equally across the video duration

## 🆘 Troubleshooting

**Issue**: Bounding boxes disappear after drawing
- **Solution**: This has been fixed - boxes now appear immediately after drawing

**Issue**: Cannot zoom out fully
- **Solution**: Minimum zoom is calculated to fit the entire image in the canvas

**Issue**: AttributeError on loading videos
- **Solution**: This has been fixed - annotations are loaded after display_width is set

**Issue**: Progress not saving
- **Solution**: Press Enter or use "Next Video" button (not just closing the app)

## 📜 Version History

- **v1.0**: Initial release with basic annotation features
- **v1.1**: Added zoom and pan functionality
- **v1.2**: Improved UI with modern styling
- **v1.3**: Added progress tracking and auto-resume
- **v1.4**: Added status textbox (removed blocking dialogs)
- **v1.5**: Added Previous Video button and detection preview zoom
- **v1.6**: Fixed bounding box disappearing bug and coordinate loading issues
