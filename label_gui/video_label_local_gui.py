import os
import csv
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading

################################################################################
# Constants
################################################################################

CLASSES = ["Bat", "Bird", "DragonFly", "Drone", "Plane", "Other"]
CLASS_MAPPING = {name: idx for idx, name in enumerate(CLASSES)}

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

def extract_frames_from_video(video_path, num_frames=10):
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
        # Equally distribute frames across the video
        step = (total_frames - 1) / (num_frames - 1) if num_frames > 1 else 0
        indices = [int(i * step) for i in range(num_frames)]
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, frame))
    
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

################################################################################
# Video Annotation GUI
################################################################################

class VideoAnnotatorGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Video Annotation Tool for YOLO Training")
        self.master.geometry("2200x1150")
        
        # Variables
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.num_frames = tk.IntVar(value=10)
        
        self.video_files = []  # List of video basenames
        self.current_video_idx = 0
        self.current_frame_idx = 0
        self.extracted_frames = []  # List of (frame_number, frame_image)
        self.current_class = tk.StringVar(value="Bird")
        
        # Bounding boxes for current frame: list of (class_id, x1, y1, x2, y2)
        self.current_bboxes = []
        self.drawing = False
        self.start_x = self.start_y = 0
        self.temp_rect = None
        
        self.detection_img_path = None
        self.video_path = None
        
        # Zoom variables
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the GUI layout"""
        # Top frame for directory selection
        top_frame = tk.Frame(self.master, padx=10, pady=10)
        top_frame.pack(fill=tk.X)
        
        tk.Label(top_frame, text="Input Directory:").grid(row=0, column=0, sticky=tk.W)
        tk.Entry(top_frame, textvariable=self.input_dir, width=50).grid(row=0, column=1, padx=5)
        tk.Button(top_frame, text="Browse", command=self.browse_input).grid(row=0, column=2)
        
        tk.Label(top_frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W)
        tk.Entry(top_frame, textvariable=self.output_dir, width=50).grid(row=1, column=1, padx=5)
        tk.Button(top_frame, text="Browse", command=self.browse_output).grid(row=1, column=2)
        
        tk.Label(top_frame, text="Number of Frames:").grid(row=2, column=0, sticky=tk.W)
        tk.Spinbox(top_frame, from_=1, to=100, textvariable=self.num_frames, width=10).grid(row=2, column=1, sticky=tk.W, padx=5)
        
        tk.Button(top_frame, text="Load Videos", command=self.load_videos, bg="green", fg="white").grid(row=3, column=1, pady=10)
        
        # Main content frame
        content_frame = tk.Frame(self.master)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        
        # Left panel - Detection image and video player
        left_panel = tk.Frame(content_frame, width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        tk.Label(left_panel, text="Detection Preview:", font=("Arial", 10, "bold")).pack()
        self.detection_canvas = tk.Canvas(left_panel, width=350, height=250, bg="gray")
        self.detection_canvas.pack(pady=5)
        
        tk.Label(left_panel, text="Video Player:", font=("Arial", 10, "bold")).pack()
        self.video_canvas = tk.Canvas(left_panel, width=350, height=250, bg="black")
        self.video_canvas.pack(pady=5)
        
        video_controls = tk.Frame(left_panel)
        video_controls.pack(pady=5)
        
        self.play_btn = tk.Button(video_controls, text="▶ Play", command=self.play_video)
        self.play_btn.grid(row=0, column=0, padx=2)
        
        self.pause_btn = tk.Button(video_controls, text="⏸ Pause", command=self.pause_video)
        self.pause_btn.grid(row=0, column=1, padx=2)
        
        tk.Button(video_controls, text="Show Frame Time", command=self.show_frame_time).grid(row=0, column=2, padx=2)
        
        self.video_label = tk.Label(left_panel, text="Video: N/A", wraplength=350)
        self.video_label.pack(pady=5)
        
        # Right panel - Frame annotation
        right_panel = tk.Frame(content_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # Info label
        self.info_label = tk.Label(right_panel, text="Load videos to begin annotation", font=("Arial", 12))
        self.info_label.pack()
        
        # Frame canvas
        tk.Label(right_panel, text="Current Frame (Click & Drag to annotate):", font=("Arial", 10, "bold")).pack()
        self.canvas_width = 1920
        self.canvas_height = 1080
        self.frame_canvas = tk.Canvas(right_panel, width=self.canvas_width, height=self.canvas_height, bg="gray")
        self.frame_canvas.pack(pady=5)
        
        # Bind mouse events for drawing bounding boxes
        self.frame_canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.frame_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.frame_canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        # Bind middle mouse button for panning
        self.frame_canvas.bind("<ButtonPress-2>", self.on_pan_start)
        self.frame_canvas.bind("<B2-Motion>", self.on_pan_drag)
        self.frame_canvas.bind("<ButtonRelease-2>", self.on_pan_end)
        
        # Bind mouse wheel for zooming
        self.frame_canvas.bind("<MouseWheel>", self.on_zoom)
        self.frame_canvas.bind("<Button-4>", self.on_zoom)  # Linux scroll up
        self.frame_canvas.bind("<Button-5>", self.on_zoom)  # Linux scroll down
        
        # Bind keyboard events for navigation
        self.master.bind("<Left>", self.prev_frame)
        self.master.bind("<Right>", self.next_frame)
        self.master.bind("<Delete>", self.delete_last_bbox)
        self.master.bind("<BackSpace>", self.delete_last_bbox)
        
        # Controls
        controls_frame = tk.Frame(right_panel)
        controls_frame.pack(pady=10)
        
        tk.Label(controls_frame, text="Class:").grid(row=0, column=0, padx=5)
        class_dropdown = ttk.Combobox(controls_frame, textvariable=self.current_class, 
                                       values=CLASSES, state="readonly", width=15)
        class_dropdown.grid(row=0, column=1, padx=5)
        
        tk.Button(controls_frame, text="◀ Prev Frame (←)", command=lambda: self.prev_frame(None)).grid(row=0, column=2, padx=5)
        tk.Button(controls_frame, text="Next Frame (→) ▶", command=lambda: self.next_frame(None)).grid(row=0, column=3, padx=5)
        tk.Button(controls_frame, text="Delete Last Box (Del)", command=lambda: self.delete_last_bbox(None)).grid(row=0, column=4, padx=5)
        
        tk.Button(controls_frame, text="Reset Zoom", command=self.reset_zoom).grid(row=1, column=0, pady=5)
        tk.Button(controls_frame, text="Next Video", command=self.next_video, bg="blue", fg="white").grid(row=1, column=1, pady=5)
        tk.Button(controls_frame, text="Save All & Exit", command=self.save_and_exit, bg="red", fg="white").grid(row=1, column=3, pady=5)
        
        # Zoom info label
        self.zoom_label = tk.Label(controls_frame, text="Zoom: 100% (Use mouse wheel)")
        self.zoom_label.grid(row=1, column=4, padx=5)
        
        # Video playback
        self.video_playing = False
        self.video_cap = None
        
    def browse_input(self):
        """Browse for input directory"""
        directory = filedialog.askdirectory()
        if directory:
            self.input_dir.set(directory)
            # Auto-fill output directory
            self.output_dir.set(get_mirror_output_path(directory))
    
    def browse_output(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir.set(directory)
    
    def load_videos(self):
        """Load all videos from input directory"""
        input_path = self.input_dir.get()
        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("Error", "Please select a valid input directory")
            return
        
        # Find all video files
        all_files = os.listdir(input_path)
        video_files = [f for f in all_files if f.lower().endswith('.mp4')]
        
        if not video_files:
            messagebox.showerror("Error", "No video files found in the input directory")
            return
        
        # Extract basenames (without extension)
        self.video_files = sorted([extract_basename(f) for f in video_files])
        self.current_video_idx = 0
        
        # Create output directory structure
        output_path = self.output_dir.get()
        ensure_dir(output_path)
        ensure_dir(os.path.join(output_path, "images"))
        ensure_dir(os.path.join(output_path, "labels"))
        
        # Save class mapping CSV
        self.save_class_mapping()
        
        # Load first video
        self.load_current_video()
        
        messagebox.showinfo("Success", f"Loaded {len(self.video_files)} video(s)")
    
    def save_class_mapping(self):
        """Save class mapping to CSV file"""
        output_path = self.output_dir.get()
        csv_path = os.path.join(output_path, "classes.csv")
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["class_id", "class_name"])
            for class_name, class_id in CLASS_MAPPING.items():
                writer.writerow([class_id, class_name])
    
    def load_current_video(self):
        """Load the current video and extract frames"""
        if not self.video_files:
            return
        
        basename = self.video_files[self.current_video_idx]
        input_path = self.input_dir.get()
        
        self.video_path = os.path.join(input_path, basename + ".mp4")
        self.detection_img_path = os.path.join(input_path, basename + ".jpg")
        
        # Check if files exist
        if not os.path.exists(self.video_path):
            messagebox.showerror("Error", f"Video not found: {self.video_path}")
            return
        
        # Extract frames
        self.extracted_frames = extract_frames_from_video(self.video_path, self.num_frames.get())
        
        if not self.extracted_frames:
            messagebox.showerror("Error", f"Could not extract frames from: {self.video_path}")
            return
        
        self.current_frame_idx = 0
        
        # Reset zoom when loading new video
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        
        # Update UI
        self.update_info_label()
        self.display_detection_image()
        self.load_video_player()
        self.display_current_frame()
    
    def update_info_label(self):
        """Update the info label with current video/frame info"""
        video_num = self.current_video_idx + 1
        total_videos = len(self.video_files)
        frame_num = self.current_frame_idx + 1
        total_frames = len(self.extracted_frames)
        
        basename = self.video_files[self.current_video_idx]
        self.info_label.config(
            text=f"Video {video_num}/{total_videos}: {basename} | Frame {frame_num}/{total_frames}"
        )
        self.video_label.config(text=f"Video: {basename}.mp4")
    
    def display_detection_image(self):
        """Display the detection preview image"""
        if os.path.exists(self.detection_img_path):
            img = Image.open(self.detection_img_path)
            img = img.resize((350, 250), Image.Resampling.LANCZOS)
            self.detection_photo = ImageTk.PhotoImage(img)
            self.detection_canvas.delete("all")
            self.detection_canvas.create_image(0, 0, anchor=tk.NW, image=self.detection_photo)
        else:
            self.detection_canvas.delete("all")
            self.detection_canvas.create_text(175, 125, text="No detection image", fill="white")
    
    def load_video_player(self):
        """Initialize video player"""
        if self.video_cap:
            self.video_cap.release()
        
        self.video_cap = cv2.VideoCapture(self.video_path)
        self.video_playing = False
        
        # Display first frame
        ret, frame = self.video_cap.read()
        if ret:
            self.display_video_frame(frame)
    
    def display_video_frame(self, frame):
        """Display a frame in the video canvas"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((350, 250), Image.Resampling.LANCZOS)
        self.video_photo = ImageTk.PhotoImage(img)
        self.video_canvas.delete("all")
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.video_photo)
    
    def play_video(self):
        """Play the video"""
        if not self.video_cap:
            return
        
        self.video_playing = True
        threading.Thread(target=self._play_video_thread, daemon=True).start()
    
    def _play_video_thread(self):
        """Thread function to play video"""
        fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps) if fps > 0 else 33
        
        while self.video_playing:
            ret, frame = self.video_cap.read()
            if not ret:
                # Loop back to start
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.video_cap.read()
            
            if ret:
                self.master.after(0, self.display_video_frame, frame)
            
            self.master.after(delay)
    
    def pause_video(self):
        """Pause the video"""
        self.video_playing = False
    
    def show_frame_time(self):
        """Show the timestamp of the current frame in the video"""
        if not self.extracted_frames:
            return
        
        frame_number, _ = self.extracted_frames[self.current_frame_idx]
        timestamp = get_frame_timestamp(self.video_path, frame_number)
        
        messagebox.showinfo("Frame Timestamp", 
                           f"Current frame occurs at: {timestamp}")
    
    def display_current_frame(self):
        """Display the current frame for annotation"""
        if not self.extracted_frames:
            return
        
        frame_number, frame = self.extracted_frames[self.current_frame_idx]
        
        # Load existing annotations if they exist
        self.load_existing_annotations()
        
        # Convert frame to PhotoImage
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_frame_original = frame_rgb.copy()
        self.frame_height, self.frame_width = frame_rgb.shape[:2]
        
        img = Image.fromarray(frame_rgb)
        # Resize to fit canvas while maintaining aspect ratio
        display_width = 1920
        display_height = int(self.frame_height * display_width / self.frame_width)
        if display_height > 1080:
            display_height = 1080
            display_width = int(self.frame_width * display_height / self.frame_height)
        
        self.display_width = display_width
        self.display_height = display_height
        
        # Calculate minimum zoom level to fit image in canvas
        self.min_zoom = max(
            self.canvas_width / self.display_width,
            self.canvas_height / self.display_height
        )
        
        # Store original display image
        self.base_display_img = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
        
        # Apply zoom and pan
        self.update_zoomed_image()
        
        self.update_info_label()
    
    def load_existing_annotations(self):
        """Load existing annotations for the current frame"""
        basename = self.video_files[self.current_video_idx]
        frame_number, _ = self.extracted_frames[self.current_frame_idx]
        frame_name = f"{basename}_{frame_number}"
        
        output_path = self.output_dir.get()
        label_path = os.path.join(output_path, "labels", f"{frame_name}.txt")
        
        self.current_bboxes = []
        
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
                        
                        # Convert normalized coords to pixel coords (display size)
                        x1 = int((x_center - width/2) * self.display_width)
                        y1 = int((y_center - height/2) * self.display_height)
                        x2 = int((x_center + width/2) * self.display_width)
                        y2 = int((y_center + height/2) * self.display_height)
                        
                        self.current_bboxes.append((class_id, x1, y1, x2, y2))
    
    def redraw_bboxes(self):
        """Redraw all bounding boxes on the canvas"""
        colors = ["red", "blue", "green", "yellow", "purple", "orange"]
        
        for class_id, x1, y1, x2, y2 in self.current_bboxes:
            # Convert image coords to canvas coords with zoom
            canvas_x1, canvas_y1 = self.image_to_canvas_coords(x1, y1)
            canvas_x2, canvas_y2 = self.image_to_canvas_coords(x2, y2)
            
            color = colors[class_id % len(colors)]
            self.frame_canvas.create_rectangle(canvas_x1, canvas_y1, canvas_x2, canvas_y2, outline=color, width=2)
            # Add class label
            class_name = CLASSES[class_id]
            self.frame_canvas.create_text(canvas_x1, canvas_y1-5, text=class_name, fill=color, anchor=tk.SW)
    
    def canvas_to_image_coords(self, canvas_x, canvas_y):
        """Convert canvas coordinates to image coordinates accounting for zoom and pan"""
        img_x = (canvas_x - self.pan_x) / self.zoom_level
        img_y = (canvas_y - self.pan_y) / self.zoom_level
        return img_x, img_y
    
    def image_to_canvas_coords(self, img_x, img_y):
        """Convert image coordinates to canvas coordinates accounting for zoom and pan"""
        canvas_x = img_x * self.zoom_level + self.pan_x
        canvas_y = img_y * self.zoom_level + self.pan_y
        return canvas_x, canvas_y
    
    def update_zoomed_image(self):
        """Update the displayed image with current zoom and pan"""
        if not hasattr(self, 'base_display_img'):
            return
        
        # Calculate zoomed image size
        zoomed_width = int(self.display_width * self.zoom_level)
        zoomed_height = int(self.display_height * self.zoom_level)
        
        # Resize image with zoom
        zoomed_img = self.base_display_img.resize((zoomed_width, zoomed_height), Image.Resampling.LANCZOS)
        
        self.frame_photo = ImageTk.PhotoImage(zoomed_img)
        
        self.frame_canvas.delete("all")
        self.frame_canvas.create_image(self.pan_x, self.pan_y, anchor=tk.NW, image=self.frame_photo)
        
        # Draw existing bounding boxes
        self.redraw_bboxes()
        
        # Update zoom label
        if hasattr(self, 'zoom_label'):
            self.zoom_label.config(text=f"Zoom: {int(self.zoom_level * 100)}% (Mouse wheel)")
    
    def on_zoom(self, event):
        """Handle mouse wheel zoom"""
        # Get mouse position
        mouse_x = event.x
        mouse_y = event.y
        
        # Convert to image coordinates before zoom
        img_x, img_y = self.canvas_to_image_coords(mouse_x, mouse_y)
        
        # Determine zoom direction
        if event.num == 5 or event.delta < 0:
            # Zoom out
            zoom_factor = 0.9
        else:
            # Zoom in
            zoom_factor = 1.1
        
        # Update zoom level with constraints
        new_zoom = self.zoom_level * zoom_factor
        min_allowed_zoom = self.min_zoom if hasattr(self, 'min_zoom') else 0.1
        if min_allowed_zoom <= new_zoom <= 10.0:
            self.zoom_level = new_zoom
            
            # Adjust pan to keep mouse position stable
            new_canvas_x, new_canvas_y = self.image_to_canvas_coords(img_x, img_y)
            self.pan_x += mouse_x - new_canvas_x
            self.pan_y += mouse_y - new_canvas_y
            
            self.update_zoomed_image()
    
    def reset_zoom(self):
        """Reset zoom and pan to default"""
        self.zoom_level = self.min_zoom if hasattr(self, 'min_zoom') else 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.update_zoomed_image()
    
    def on_pan_start(self, event):
        """Start panning with middle mouse button"""
        self.panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
    
    def on_pan_drag(self, event):
        """Pan the image"""
        if self.panning:
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            self.pan_x += dx
            self.pan_y += dy
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            self.update_zoomed_image()
    
    def on_pan_end(self, event):
        """End panning"""
        self.panning = False
    
    def on_mouse_down(self, event):
        """Handle mouse button press"""
        self.drawing = True
        self.start_x = event.x
        self.start_y = event.y
    
    def on_mouse_drag(self, event):
        """Handle mouse drag"""
        if not self.drawing:
            return
        
        if self.temp_rect:
            self.frame_canvas.delete(self.temp_rect)
        
        self.temp_rect = self.frame_canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y, 
            outline="red", width=2
        )
    
    def on_mouse_up(self, event):
        """Handle mouse button release"""
        if not self.drawing:
            return
        
        self.drawing = False
        
        # Convert canvas coordinates to image coordinates
        img_x1, img_y1 = self.canvas_to_image_coords(self.start_x, self.start_y)
        img_x2, img_y2 = self.canvas_to_image_coords(event.x, event.y)
        
        # Ensure x1 < x2 and y1 < y2
        if img_x1 > img_x2:
            img_x1, img_x2 = img_x2, img_x1
        if img_y1 > img_y2:
            img_y1, img_y2 = img_y2, img_y1
        
        # Constrain to image boundaries
        img_x1 = max(0, min(img_x1, self.display_width))
        img_y1 = max(0, min(img_y1, self.display_height))
        img_x2 = max(0, min(img_x2, self.display_width))
        img_y2 = max(0, min(img_y2, self.display_height))
        
        # Ignore very small boxes (in image coordinates)
        if abs(img_x2 - img_x1) < 5 or abs(img_y2 - img_y1) < 5:
            if self.temp_rect:
                self.frame_canvas.delete(self.temp_rect)
                self.temp_rect = None
            return
        
        # Add bounding box in image coordinates
        class_id = CLASS_MAPPING[self.current_class.get()]
        self.current_bboxes.append((class_id, int(img_x1), int(img_y1), int(img_x2), int(img_y2)))
        
        if self.temp_rect:
            self.frame_canvas.delete(self.temp_rect)
            self.temp_rect = None
        
        # Redraw all boxes
        self.update_zoomed_image()
    
    def delete_last_bbox(self, event):
        """Delete the last bounding box"""
        if self.current_bboxes:
            self.current_bboxes.pop()
            self.update_zoomed_image()
    
    def prev_frame(self, event):
        """Go to previous frame"""
        self.save_current_frame_annotations()
        
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.display_current_frame()
    
    def next_frame(self, event):
        """Go to next frame"""
        self.save_current_frame_annotations()
        
        if self.current_frame_idx < len(self.extracted_frames) - 1:
            self.current_frame_idx += 1
            self.display_current_frame()
        else:
            messagebox.showinfo("Info", "Last frame of this video. Use 'Next Video' to continue.")
    
    def next_video(self):
        """Move to the next video"""
        self.save_current_frame_annotations()
        
        if self.current_video_idx < len(self.video_files) - 1:
            self.current_video_idx += 1
            self.load_current_video()
        else:
            messagebox.showinfo("Info", "All videos processed!")
    
    def save_current_frame_annotations(self):
        """Save annotations for the current frame"""
        if not self.extracted_frames:
            return
        
        basename = self.video_files[self.current_video_idx]
        frame_number, frame = self.extracted_frames[self.current_frame_idx]
        frame_name = f"{basename}_{frame_number}"
        
        output_path = self.output_dir.get()
        images_dir = os.path.join(output_path, "images")
        labels_dir = os.path.join(output_path, "labels")
        
        # Save frame image
        img_path = os.path.join(images_dir, f"{frame_name}.jpg")
        cv2.imwrite(img_path, frame)
        
        # Save annotations if any exist
        if self.current_bboxes:
            label_path = os.path.join(labels_dir, f"{frame_name}.txt")
            with open(label_path, 'w') as f:
                for class_id, x1, y1, x2, y2 in self.current_bboxes:
                    # Convert to normalized YOLO format
                    x_center = ((x1 + x2) / 2) / self.display_width
                    y_center = ((y1 + y2) / 2) / self.display_height
                    width = abs(x2 - x1) / self.display_width
                    height = abs(y2 - y1) / self.display_height
                    
                    # Constrain normalized values to [0, 1]
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    width = max(0.0, min(1.0, width))
                    height = max(0.0, min(1.0, height))
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        else:
            # No annotations - delete label file if it exists
            label_path = os.path.join(labels_dir, f"{frame_name}.txt")
            if os.path.exists(label_path):
                os.remove(label_path)
    
    def save_and_exit(self):
        """Save all and exit"""
        self.save_current_frame_annotations()
        
        # Save all remaining frames from all videos
        for video_idx in range(len(self.video_files)):
            basename = self.video_files[video_idx]
            video_path = os.path.join(self.input_dir.get(), basename + ".mp4")
            
            frames = extract_frames_from_video(video_path, self.num_frames.get())
            output_path = self.output_dir.get()
            images_dir = os.path.join(output_path, "images")
            
            for frame_number, frame in frames:
                frame_name = f"{basename}_{frame_number}"
                img_path = os.path.join(images_dir, f"{frame_name}.jpg")
                
                # Only save if doesn't exist (to preserve user annotations)
                if not os.path.exists(img_path):
                    cv2.imwrite(img_path, frame)
        
        messagebox.showinfo("Success", "All annotations saved successfully!")
        self.master.quit()

################################################################################
# Main Entry Point
################################################################################

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoAnnotatorGUI(root)
    root.mainloop()
