# TODO: Add possibility to add camera name or ID in the output file names

import os
import csv
import json
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading
from tkinter import font as tkfont

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
        
        # Modern color scheme
        self.colors = {
            'bg_primary': '#ffffff',
            'bg_secondary': '#f8f9fa',
            'bg_tertiary': '#e9ecef',
            'primary': '#3b82f6',
            'primary_dark': '#1e40af',
            'success': '#10b981',
            'danger': '#ef4444',
            'text_primary': '#1f2937',
            'text_secondary': '#6b7280',
            'border': '#d1d5db'
        }
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.setup_styles()
        
        # Configure main window
        self.master.configure(bg=self.colors['bg_secondary'])
        
        # Variables
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.num_frames = tk.IntVar(value=10)
        self.frame_increment_mode = tk.StringVar(value="stride")  # "stride", "distributed"
        self.frame_inclusion_count = tk.IntVar(value=5)
        self.frame_inclusion_stride = tk.IntVar(value=5)
        self.frame_inclusion_direction = tk.StringVar(value="next")  # "next", "prev"
        self.frame_distributed_scope = tk.StringVar(value="all")  # "all", "missing"
        
        self.video_files = []  # List of video basenames
        self.current_video_idx = 0
        self.current_frame_idx = 0
        self.extracted_frames = []  # List of (frame_number, frame_image)
        self.current_class = tk.StringVar(value="Bird")
        
        # Per-video frame count overrides: {video_basename: custom_frame_count}
        self.video_frame_counts = {}
        
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
        
        # Detection image zoom and pan state
        self.detection_zoom = 1.0
        self.detection_img_original = None
        self.detection_pan_x = 0
        self.detection_pan_y = 0
        self.detection_panning = False
        self.detection_pan_start_x = 0
        self.detection_pan_start_y = 0
        
        # Image cache for zoomed display to avoid repeated PIL resizing
        self.cached_zoomed_img = None
        self.cached_zoom_level = None
        
        self.setup_ui()
    
    def setup_styles(self):
        """Configure custom ttk styles"""
        # Button styles
        self.style.configure('Primary.TButton',
                           font=('Segoe UI', 10, 'bold'),
                           padding=8,
                           background=self.colors['primary'],
                           foreground='white')
        
        self.style.configure('Success.TButton',
                           font=('Segoe UI', 10, 'bold'),
                           padding=8,
                           background=self.colors['success'],
                           foreground='white')
        
        self.style.configure('Danger.TButton',
                           font=('Segoe UI', 10, 'bold'),
                           padding=8,
                           background=self.colors['danger'],
                           foreground='white')
        
        # Label styles
        self.style.configure('Title.TLabel',
                           font=('Segoe UI', 14, 'bold'),
                           background=self.colors['bg_primary'],
                           foreground=self.colors['primary'])
        
        self.style.configure('Header.TLabel',
                           font=('Segoe UI', 11, 'bold'),
                           background=self.colors['bg_primary'],
                           foreground=self.colors['text_primary'])
        
        # Frame styles
        self.style.configure('Card.TFrame',
                           background=self.colors['bg_primary'],
                           relief='flat',
                           borderwidth=0)
        
        # Combobox
        self.style.configure('TCombobox',
                           fieldbackground=self.colors['bg_primary'],
                           background=self.colors['bg_primary'])
        
        # Spinbox
        self.style.configure('TSpinbox',
                           fieldbackground=self.colors['bg_primary'],
                           background=self.colors['bg_primary'])
        
        self.style.map('TCombobox',
                     fieldbackground=[('readonly', self.colors['bg_primary'])])
        
        # Entry
        self.style.configure('TEntry',
                           fieldbackground=self.colors['bg_primary'],
                           background=self.colors['bg_primary'])
        
    def setup_ui(self):
        """Setup the GUI layout"""
        # Header frame
        header_frame = tk.Frame(self.master, bg=self.colors['primary'], height=80)
        header_frame.pack(fill=tk.X, padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        header_title = tk.Label(
            header_frame,
            text="🎬 Video-frames annotator",
            font=('Segoe UI', 16, 'bold'),
            bg=self.colors['primary'],
            fg='white'
        )
        header_title.pack(pady=10)
        
        header_subtitle = tk.Label(
            header_frame,
            text="Draw a bounding box around each target in the video frames",
            font=('Segoe UI', 9),
            bg=self.colors['primary'],
            fg='white'
        )
        header_subtitle.pack()
        
        # Top frame for directory selection
        top_frame = tk.Frame(self.master, bg=self.colors['bg_secondary'])
        top_frame.pack(fill=tk.X, padx=15, pady=15)
        
        # Input directory section
        input_label = tk.Label(top_frame, text="📁 Input Directory:", 
                              font=('Segoe UI', 10, 'bold'),
                              bg=self.colors['bg_secondary'],
                              fg=self.colors['text_primary'])
        input_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        input_entry = tk.Entry(top_frame, textvariable=self.input_dir, 
                              font=('Segoe UI', 10),
                              bg=self.colors['bg_primary'],
                              fg=self.colors['text_primary'],
                              relief=tk.FLAT,
                              width=50,
                              bd=0)
        input_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        input_entry.config(highlightthickness=1, highlightcolor=self.colors['border'])
        
        browse_input_btn = tk.Button(top_frame, text="📂 Browse", 
                                    command=self.browse_input,
                                    font=('Segoe UI', 9, 'bold'),
                                    bg=self.colors['primary'],
                                    fg='white',
                                    relief=tk.FLAT,
                                    bd=0,
                                    padx=15,
                                    pady=6,
                                    cursor='hand2',
                                    activebackground=self.colors['primary_dark'])
        browse_input_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # Output directory section
        output_label = tk.Label(top_frame, text="💾 Output Directory:", 
                               font=('Segoe UI', 10, 'bold'),
                               bg=self.colors['bg_secondary'],
                               fg=self.colors['text_primary'])
        output_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        output_entry = tk.Entry(top_frame, textvariable=self.output_dir, 
                               font=('Segoe UI', 10),
                               bg=self.colors['bg_primary'],
                               fg=self.colors['text_primary'],
                               relief=tk.FLAT,
                               width=50,
                               bd=0)
        output_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        output_entry.config(highlightthickness=1, highlightcolor=self.colors['border'])
        
        browse_output_btn = tk.Button(top_frame, text="📂 Browse", 
                                     command=self.browse_output,
                                     font=('Segoe UI', 9, 'bold'),
                                     bg=self.colors['primary'],
                                     fg='white',
                                     relief=tk.FLAT,
                                     bd=0,
                                     padx=15,
                                     pady=6,
                                     cursor='hand2',
                                     activebackground=self.colors['primary_dark'])
        browse_output_btn.grid(row=1, column=2, padx=5, pady=5)
        
        # Frames setting
        frames_label = tk.Label(top_frame, text="🎞️ Frames per Video:", 
                               font=('Segoe UI', 10, 'bold'),
                               bg=self.colors['bg_secondary'],
                               fg=self.colors['text_primary'])
        frames_label.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        
        frames_spinbox = tk.Spinbox(top_frame, from_=1, to=100, 
                                   textvariable=self.num_frames,
                                   font=('Segoe UI', 10),
                                   bg=self.colors['bg_primary'],
                                   fg=self.colors['text_primary'],
                                   relief=tk.FLAT,
                                   width=15,
                                   bd=0)
        frames_spinbox.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        load_btn = tk.Button(top_frame, text="🚀 Load Videos", 
                            command=self.load_videos,
                            font=('Segoe UI', 10, 'bold'),
                            bg=self.colors['success'],
                            fg='white',
                            relief=tk.FLAT,
                            bd=0,
                            padx=20,
                            pady=8,
                            cursor='hand2',
                            activebackground='#059669')
        load_btn.grid(row=2, column=2, padx=5, pady=5)
        
        top_frame.columnconfigure(1, weight=1)
        
        # Main content frame
        content_frame = tk.Frame(self.master, bg=self.colors['bg_secondary'])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Detection image and video player
        left_panel = tk.Frame(content_frame, bg=self.colors['bg_secondary'], width=380)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        left_panel.pack_propagate(False)
        
        # Detection card
        detection_card = tk.Frame(left_panel, bg=self.colors['bg_primary'], relief=tk.FLAT)
        detection_card.pack(fill=tk.X, pady=5)
        
        detection_title = tk.Label(detection_card, text="📸 Detection Preview", 
                                  font=('Segoe UI', 10, 'bold'),
                                  bg=self.colors['bg_primary'],
                                  fg=self.colors['primary'])
        detection_title.pack(pady=(10, 5), padx=10)
        
        self.detection_canvas = tk.Canvas(detection_card, width=350, height=250, 
                                         bg=self.colors['bg_tertiary'],
                                         relief=tk.FLAT,
                                         bd=0,
                                         highlightthickness=0)
        self.detection_canvas.pack(pady=5, padx=10)
        
        # Bind mouse wheel for detection image zoom
        self.detection_canvas.bind("<MouseWheel>", self.on_detection_zoom)
        self.detection_canvas.bind("<Button-4>", self.on_detection_zoom)  # Linux scroll up
        self.detection_canvas.bind("<Button-5>", self.on_detection_zoom)  # Linux scroll down
        self.detection_canvas.bind("<Double-Button-1>", self.reset_detection_zoom)  # Double-click to reset
        
        # Bind middle-click for detection panning
        self.detection_canvas.bind("<ButtonPress-2>", self.on_detection_pan_start)
        self.detection_canvas.bind("<B2-Motion>", self.on_detection_pan_drag)
        self.detection_canvas.bind("<ButtonRelease-2>", self.on_detection_pan_end)
        
        # Video card
        video_card = tk.Frame(left_panel, bg=self.colors['bg_primary'], relief=tk.FLAT)
        video_card.pack(fill=tk.X, pady=5)
        
        video_title = tk.Label(video_card, text="🎬 Video Player", 
                              font=('Segoe UI', 10, 'bold'),
                              bg=self.colors['bg_primary'],
                              fg=self.colors['primary'])
        video_title.pack(pady=(10, 5), padx=10)
        
        self.video_canvas = tk.Canvas(video_card, width=350, height=250, 
                                     bg='#000000',
                                     relief=tk.FLAT,
                                     bd=0,
                                     highlightthickness=0)
        self.video_canvas.pack(pady=5, padx=10)
        
        video_controls = tk.Frame(video_card, bg=self.colors['bg_primary'])
        video_controls.pack(pady=10, padx=10, fill=tk.X)
        
        self.play_btn = tk.Button(video_controls, text="▶ Play", 
                                 command=self.play_video,
                                 font=('Segoe UI', 9, 'bold'),
                                 bg=self.colors['primary'],
                                 fg='white',
                                 relief=tk.FLAT,
                                 bd=0,
                                 padx=10,
                                 pady=5,
                                 cursor='hand2',
                                 activebackground=self.colors['primary_dark'])
        self.play_btn.grid(row=0, column=0, padx=2)
        
        self.pause_btn = tk.Button(video_controls, text="⏸ Pause", 
                                  command=self.pause_video,
                                  font=('Segoe UI', 9, 'bold'),
                                  bg=self.colors['primary'],
                                  fg='white',
                                  relief=tk.FLAT,
                                  bd=0,
                                  padx=10,
                                  pady=5,
                                  cursor='hand2',
                                  activebackground=self.colors['primary_dark'])
        self.pause_btn.grid(row=0, column=1, padx=2)
        
        time_btn = tk.Button(video_controls, text="⏱️ Frame Time", 
                            command=self.show_frame_time,
                            font=('Segoe UI', 9, 'bold'),
                            bg=self.colors['primary'],
                            fg='white',
                            relief=tk.FLAT,
                            bd=0,
                            padx=10,
                            pady=5,
                            cursor='hand2',
                            activebackground=self.colors['primary_dark'])
        time_btn.grid(row=0, column=2, padx=2)
        
        video_controls.columnconfigure(0, weight=1)
        video_controls.columnconfigure(1, weight=1)
        video_controls.columnconfigure(2, weight=1)
        
        self.video_label = tk.Label(video_card, text="Video: N/A", 
                                   font=('Segoe UI', 8),
                                   bg=self.colors['bg_primary'],
                                   fg=self.colors['text_secondary'],
                                   wraplength=350)
        self.video_label.pack(pady=5, padx=10)
        
        # Right panel - Frame annotation
        right_panel = tk.Frame(content_frame, bg=self.colors['bg_secondary'])
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # Info label
        self.info_label = tk.Label(right_panel, text="Load videos to begin annotation", 
                                  font=('Segoe UI', 11, 'bold'),
                                  bg=self.colors['bg_secondary'],
                                  fg=self.colors['primary'])
        self.info_label.pack(pady=10)
        
        # Progress bar
        progress_frame = tk.Frame(right_panel, bg=self.colors['bg_secondary'])
        progress_frame.pack(pady=(0, 10), padx=20, fill=tk.X)
        
        self.progress_label = tk.Label(progress_frame, text="Overall Progress: 0%", 
                                       font=('Segoe UI', 9),
                                       bg=self.colors['bg_secondary'],
                                       fg=self.colors['text_secondary'])
        self.progress_label.pack()
        
        self.progress_bar = ttk.Progressbar(progress_frame, orient='horizontal', 
                                           length=400, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Status textbox for non-blocking messages
        status_frame = tk.Frame(right_panel, bg=self.colors['bg_secondary'])
        status_frame.pack(pady=(0, 10), padx=20, fill=tk.X)
        
        status_label = tk.Label(status_frame, text="Status:", 
                               font=('Segoe UI', 9, 'bold'),
                               bg=self.colors['bg_secondary'],
                               fg=self.colors['text_primary'])
        status_label.pack(anchor=tk.W)
        
        self.status_text = tk.Text(status_frame, height=2, width=50, 
                                   font=('Segoe UI', 9),
                                   bg='#f8f9fa', fg='#212529',
                                   relief=tk.FLAT, borderwidth=1,
                                   wrap=tk.WORD, state=tk.DISABLED)
        self.status_text.pack(fill=tk.X)
        
        # Frame canvas
        canvas_label = tk.Label(right_panel, text="Current Frame (Click & Drag to annotate):", 
                               font=('Segoe UI', 10, 'bold'),
                               bg=self.colors['bg_secondary'],
                               fg=self.colors['text_primary'])
        canvas_label.pack()
        
        self.canvas_width = 1920 // 2
        self.canvas_height = 1080 // 2
        
        canvas_frame = tk.Frame(right_panel, bg=self.colors['bg_primary'], relief=tk.FLAT)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.frame_canvas = tk.Canvas(canvas_frame, width=self.canvas_width, 
                                     height=self.canvas_height,
                                     bg=self.colors['bg_tertiary'],
                                     relief=tk.FLAT,
                                     bd=0,
                                     highlightthickness=0)
        self.frame_canvas.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        
        # Bind mouse events for drawing bounding boxes
        self.frame_canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.frame_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.frame_canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        # Bind right-click for deleting bounding boxes
        self.frame_canvas.bind("<ButtonPress-3>", self.on_right_click)
        
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
        self.master.bind("<Return>", self.on_enter_key)  # Enter to save and next video
        
        # Controls
        controls_frame = tk.Frame(right_panel, bg=self.colors['bg_secondary'])
        controls_frame.pack(pady=10, padx=10, fill=tk.X)
        
        # Row 1: Class and buttons
        row1_frame = tk.Frame(controls_frame, bg=self.colors['bg_secondary'])
        row1_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(row1_frame, text="Class:", 
                font=('Segoe UI', 10, 'bold'),
                bg=self.colors['bg_secondary'],
                fg=self.colors['text_primary']).pack(side=tk.LEFT, padx=5)
        
        class_dropdown = ttk.Combobox(row1_frame, textvariable=self.current_class, 
                                      values=CLASSES, state="readonly", width=15,
                                      font=('Segoe UI', 10))
        class_dropdown.pack(side=tk.LEFT, padx=5)
        
        tk.Button(row1_frame, text="◀ Prev (←)", 
                 command=lambda: self.prev_frame(None),
                 font=('Segoe UI', 9, 'bold'),
                 bg=self.colors['primary'],
                 fg='white',
                 relief=tk.FLAT,
                 bd=0,
                 padx=10,
                 pady=5,
                 cursor='hand2',
                 activebackground=self.colors['primary_dark']).pack(side=tk.LEFT, padx=2)
        
        tk.Button(row1_frame, text="Next (→) ▶", 
                 command=lambda: self.next_frame(None),
                 font=('Segoe UI', 9, 'bold'),
                 bg=self.colors['primary'],
                 fg='white',
                 relief=tk.FLAT,
                 bd=0,
                 padx=10,
                 pady=5,
                 cursor='hand2',
                 activebackground=self.colors['primary_dark']).pack(side=tk.LEFT, padx=2)
        
        tk.Button(row1_frame, text="🗑️ Delete (Del)", 
                 command=lambda: self.delete_last_bbox(None),
                 font=('Segoe UI', 9, 'bold'),
                 bg=self.colors['danger'],
                 fg='white',
                 relief=tk.FLAT,
                 bd=0,
                 padx=10,
                 pady=5,
                 cursor='hand2',
                 activebackground='#dc2626').pack(side=tk.LEFT, padx=2)
        
        # Row 2: More controls
        row2_frame = tk.Frame(controls_frame, bg=self.colors['bg_secondary'])
        row2_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(row2_frame, text="🔍 Reset Zoom", 
                 command=self.reset_zoom,
                 font=('Segoe UI', 9, 'bold'),
                 bg=self.colors['primary'],
                 fg='white',
                 relief=tk.FLAT,
                 bd=0,
                 padx=10,
                 pady=5,
                 cursor='hand2',
                 activebackground=self.colors['primary_dark']).pack(side=tk.LEFT, padx=2)
        
        tk.Button(row2_frame, text="◀◀ Previous Video", 
                 command=self.prev_video,
                 font=('Segoe UI', 9, 'bold'),
                 bg=self.colors['primary'],
                 fg='white',
                 relief=tk.FLAT,
                 bd=0,
                 padx=10,
                 pady=5,
                 cursor='hand2',
                 activebackground='#0056b3').pack(side=tk.LEFT, padx=2)
        
        tk.Button(row2_frame, text="▶▶ Next Video (Enter)", 
                 command=self.save_video_and_next,
                 font=('Segoe UI', 9, 'bold'),
                 bg=self.colors['primary'],
                 fg='white',
                 relief=tk.FLAT,
                 bd=0,
                 padx=10,
                 pady=5,
                 cursor='hand2',
                 activebackground=self.colors['primary_dark']).pack(side=tk.LEFT, padx=2)
        
        tk.Button(row2_frame, text="💾 Save All & Exit", 
                 command=self.save_and_exit,
                 font=('Segoe UI', 9, 'bold'),
                 bg=self.colors['success'],
                 fg='white',
                 relief=tk.FLAT,
                 bd=0,
                 padx=15,
                 pady=5,
                 cursor='hand2',
                 activebackground='#059669').pack(side=tk.LEFT, padx=2)
        
        # Zoom info label
        self.zoom_label = tk.Label(row2_frame, text="Zoom: 100%", 
                                  font=('Segoe UI', 9, 'bold'),
                                  bg=self.colors['bg_tertiary'],
                                  fg=self.colors['primary'],
                                  relief=tk.FLAT,
                                  padx=10,
                                  pady=4)
        self.zoom_label.pack(side=tk.RIGHT, padx=5)
        
        # Row 3: Frame management
        row3_frame = tk.Frame(controls_frame, bg=self.colors['bg_secondary'])
        row3_frame.pack(fill=tk.X, pady=5)

        # --- Frame inclusion options ---
        tk.Label(row3_frame, text="Add Frames:", font=('Segoe UI', 9, 'bold'), bg=self.colors['bg_secondary']).pack(side=tk.LEFT, padx=2)
        tk.Spinbox(row3_frame, from_=1, to=100, textvariable=self.frame_inclusion_count, width=3, font=('Segoe UI', 9)).pack(side=tk.LEFT)

        # Mode radio buttons
        for mode, label in [("stride", "Stride"), ("distributed", "Distributed")]:
            tk.Radiobutton(row3_frame, text=label, variable=self.frame_increment_mode, value=mode, font=('Segoe UI', 9), bg=self.colors['bg_secondary']).pack(side=tk.LEFT, padx=2)

        # Direction radio buttons (only for stride)
        for direction, label in [("next", "Next"), ("prev", "Prev")]:
            tk.Radiobutton(row3_frame, text=label, variable=self.frame_inclusion_direction, value=direction, font=('Segoe UI', 9), bg=self.colors['bg_secondary']).pack(side=tk.LEFT, padx=1)

        # Stride spinbox (only for stride)
        tk.Label(row3_frame, text="Stride:", font=('Segoe UI', 9), bg=self.colors['bg_secondary']).pack(side=tk.LEFT, padx=1)
        tk.Spinbox(row3_frame, from_=1, to=50, textvariable=self.frame_inclusion_stride, width=2, font=('Segoe UI', 9)).pack(side=tk.LEFT)

        # Distributed scope (only for distributed)
        tk.Label(row3_frame, text="Scope:", font=('Segoe UI', 9), bg=self.colors['bg_secondary']).pack(side=tk.LEFT, padx=1)
        for scope, label in [("all", "All Video"), ("missing", "Missing Only")]:
            tk.Radiobutton(row3_frame, text=label, variable=self.frame_distributed_scope, value=scope, font=('Segoe UI', 9), bg=self.colors['bg_secondary']).pack(side=tk.LEFT, padx=1)

        # Add frames button
        tk.Button(row3_frame, text="➕ Add", command=self.increase_frames_for_video, font=('Segoe UI', 9, 'bold'), bg='#8b5cf6', fg='white', relief=tk.FLAT, bd=0, padx=10, pady=5, cursor='hand2', activebackground='#7c3aed').pack(side=tk.LEFT, padx=2)

        # Frame count display label
        self.frame_count_label = tk.Label(row3_frame, text="Frames: 10", font=('Segoe UI', 9, 'bold'), bg=self.colors['bg_secondary'], fg=self.colors['text_secondary'], relief=tk.FLAT)
        self.frame_count_label.pack(side=tk.LEFT, padx=10)

        # --- Video seek bar and explorer button ---
        seek_frame = tk.Frame(left_panel, bg=self.colors['bg_secondary'])
        seek_frame.pack(fill=tk.X, pady=(5, 0))
        self.video_seek_var = tk.DoubleVar(value=0.0)
        self.video_seek_slider = tk.Scale(seek_frame, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.video_seek_var, showvalue=0, length=340, command=self.on_video_seek, bg=self.colors['bg_secondary'])
        self.video_seek_slider.pack(side=tk.LEFT, padx=5)
        self.video_time_label = tk.Label(seek_frame, text="00:00 / 00:00", font=('Segoe UI', 9), bg=self.colors['bg_secondary'])
        self.video_time_label.pack(side=tk.LEFT, padx=2)
        # Explorer button
        tk.Button(seek_frame, text="Show in Explorer", command=self.open_video_in_explorer, font=('Segoe UI', 9), bg=self.colors['primary'], fg='white', relief=tk.FLAT, bd=0, padx=8, pady=2, cursor='hand2', activebackground=self.colors['primary_dark']).pack(side=tk.LEFT, padx=5)
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
        
        # Try to resume from last progress
        resumed = self.load_progress()
        
        # Load current video (either first or resumed)
        self.load_current_video()
        
        # Update progress bar
        self.update_progress_bar()
        
        if resumed:
            self.update_status(f"\u2705 Loaded {len(self.video_files)} video(s) | Resumed from video {self.current_video_idx + 1}", '#10b981')
        else:
            self.update_status(f"\u2705 Loaded {len(self.video_files)} video(s) | Ready to annotate!", '#10b981')
    
    def save_class_mapping(self):
        """Save class mapping to CSV file"""
        output_path = self.output_dir.get()
        csv_path = os.path.join(output_path, "classes.csv")
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["class_id", "class_name"])
            for class_name, class_id in CLASS_MAPPING.items():
                writer.writerow([class_id, class_name])
    
    def get_progress_file_path(self):
        """Get the path to the progress tracking file"""
        output_path = self.output_dir.get()
        return os.path.join(output_path, ".progress.json")
    
    def save_progress(self):
        """Save current progress to file"""
        progress_file = self.get_progress_file_path()
        progress_data = {
            "last_video_idx": self.current_video_idx,
            "total_videos": len(self.video_files),
            "video_frame_counts": self.video_frame_counts
        }
        try:
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f)
        except Exception as e:
            print(f"Error saving progress: {e}")
    
    def load_progress(self):
        """Load progress from file and resume from last video"""
        progress_file = self.get_progress_file_path()
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
                    last_idx = progress_data.get("last_video_idx", 0)
                    # Load per-video frame counts if available
                    self.video_frame_counts = progress_data.get("video_frame_counts", {})
                    # Only resume if the last index is valid
                    if 0 <= last_idx < len(self.video_files):
                        self.current_video_idx = last_idx
                        return True
            except Exception as e:
                print(f"Error loading progress: {e}")
        return False
    
    def update_progress_bar(self):
        """Update the progress bar and label"""
        if not self.video_files:
            return
        
        total_videos = len(self.video_files)
        current_video = self.current_video_idx + 1
        
        # Calculate percentage
        progress_percent = (current_video / total_videos) * 100
        
        # Update progress bar
        self.progress_bar['value'] = progress_percent
        
        # Update progress label
        self.progress_label.config(
            text=f"Overall Progress: {current_video}/{total_videos} videos ({progress_percent:.1f}%)"
        )
    
    def update_status(self, message, color='#212529'):
        """Update the status textbox with a message"""
        self.status_text.config(state=tk.NORMAL)
        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(1.0, message)
        self.status_text.tag_configure('color', foreground=color)
        self.status_text.tag_add('color', 1.0, tk.END)
        self.status_text.config(state=tk.DISABLED)
        # Auto-scroll to the end
        self.status_text.see(tk.END)
    
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
        
        # Get frame count for this specific video, or use global setting
        frame_count = self.video_frame_counts.get(basename, self.num_frames.get())

        # Preload existing saved frames for this video from disk first
        output_path = self.output_dir.get()
        images_dir = os.path.join(output_path, "images")
        existing_frames = []  # list[(frame_idx, frame_img)]
        existing_indices = set()

        if os.path.exists(images_dir):
            prefix = f"{basename}_"
            for fname in os.listdir(images_dir):
                if fname.startswith(prefix) and fname.lower().endswith('.jpg'):
                    name_wo_ext = os.path.splitext(fname)[0]
                    try:
                        idx_str = name_wo_ext.split('_')[-1]
                        idx = int(idx_str)
                    except Exception:
                        continue
                    img_path = os.path.join(images_dir, fname)
                    img = cv2.imread(img_path)
                    if img is not None:
                        existing_frames.append((idx, img))
                        existing_indices.add(idx)

        # Sort existing frames by index
        existing_frames.sort(key=lambda x: x[0])

        # Determine total frames in video
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Helper to compute equally distributed indices
        def compute_indices(n):
            if n <= 0:
                return []
            if n >= total_frames:
                return list(range(total_frames))
            step = (total_frames - 1) / (n - 1) if n > 1 else 0
            return [int(i * step) for i in range(n)]

        # If we already have enough existing frames on disk, use them
        self.extracted_frames = existing_frames[:]

        if len(self.extracted_frames) < frame_count:
            # Compute desired target indices and extract only the missing ones
            target_indices = compute_indices(frame_count)
            missing = [i for i in target_indices if i not in existing_indices]

            # Extract missing frames from the video
            cap = cv2.VideoCapture(self.video_path)
            for idx in missing:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    self.extracted_frames.append((idx, frame))
            cap.release()

            # Sort combined frames by frame index
            self.extracted_frames.sort(key=lambda x: x[0])

        if not self.extracted_frames:
            messagebox.showerror("Error", f"Could not extract or load frames for: {self.video_path}")
            return
        
        self.current_frame_idx = 0
        
        # Reset zoom when loading new video
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.detection_zoom = 1.0  # Reset detection preview zoom
        
        # Reset per-video display size lock so frames share same scaling
        self.display_size_locked = False
        self.display_width_base = None
        self.display_height_base = None

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
        
        # Update frame count label
        self.frame_count_label.config(text=f"Frames: {total_frames}")
        
        # Update progress bar
        self.update_progress_bar()
    
    def increase_frames_for_video(self):
        """Add 5 frames using selected mode: stride (spaced) or distributed (evenly spread)"""
        if not self.video_files or not self.extracted_frames:
            self.update_status("⚠️ No videos loaded", '#f59e0b')
            return
        
        basename = self.video_files[self.current_video_idx]
        
        # Get the current frame number being displayed
        current_frame_number, _ = self.extracted_frames[self.current_frame_idx]
        
        # Determine total frames available in video
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        existing_indices = {idx for idx, _ in self.extracted_frames}
        mode = self.frame_increment_mode.get()
        
        needed = []
        
        if mode == "stride":
            # Add 5 frames with spacing (stride=5) to reduce similarity
            stride = 5
            frame_num = current_frame_number + stride
            
            while len(needed) < 5 and frame_num < total_frames:
                if frame_num not in existing_indices:
                    needed.append(frame_num)
                frame_num += stride
            
            mode_desc = f"stride={stride}"
        else:  # distributed
            # Add 5 frames evenly distributed from current_frame to end of video
            remaining = total_frames - current_frame_number - 1
            if remaining <= 5:
                # Not enough frames, just take all remaining
                needed = [i for i in range(current_frame_number + 1, total_frames) 
                         if i not in existing_indices]
            else:
                # Distribute 5 frames evenly
                step = remaining // 5
                for i in range(1, 6):
                    frame_num = current_frame_number + (i * step)
                    if frame_num < total_frames and frame_num not in existing_indices:
                        needed.append(frame_num)
            
            mode_desc = "distributed"
        
        if not needed:
            self.update_status("ℹ️ No new frames available.", '#f59e0b')
            return
        
        needed = sorted(set(needed))
        
        try:
            self.update_status(f"⏳ Extracting {len(needed)} frame(s) ({mode_desc}) for {basename}...", '#0066cc')
            self.master.update()
            
            cap = cv2.VideoCapture(self.video_path)
            added = 0
            for idx in needed:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    self.extracted_frames.append((idx, frame))
                    added += 1
            cap.release()
            
            if added == 0:
                self.update_status("❌ Failed to extract additional frames", '#ef4444')
                return
            
            # Sort frames by index
            self.extracted_frames.sort(key=lambda x: x[0])
            new_frame_count = len(self.extracted_frames)
            self.video_frame_counts[basename] = new_frame_count
            
            # Save newly added frames to disk
            self.save_all_frames_of_current_video()
            
            # Refresh UI
            self.update_info_label()
            self.display_current_frame()
            
            self.update_status(f"✅ Added {added} frame(s) ({mode_desc}). Total now: {new_frame_count}", '#10b981')
        except Exception as e:
            self.update_status(f"❌ Error: {str(e)}", '#ef4444')
    
    def display_detection_image(self):
        """Display the detection preview image with zoom and pan support"""
        if os.path.exists(self.detection_img_path):
            # Load original image
            self.detection_img_original = Image.open(self.detection_img_path)
            
            # Calculate zoomed size
            base_width, base_height = 350, 250
            zoomed_width = int(base_width * self.detection_zoom)
            zoomed_height = int(base_height * self.detection_zoom)
            
            # Resize image with zoom
            img = self.detection_img_original.resize((zoomed_width, zoomed_height), Image.Resampling.LANCZOS)
            self.detection_photo = ImageTk.PhotoImage(img)
            
            self.detection_canvas.delete("all")
            # Place image with pan offset (default center if not zoomed)
            if self.detection_zoom <= 1.0:
                # If zoomed out, center the image
                canvas_center_x = 175
                canvas_center_y = 125
            else:
                # If zoomed in, use pan offset (default to center if not yet panned)
                if self.detection_pan_x == 0 and self.detection_pan_y == 0:
                    canvas_center_x = 175
                    canvas_center_y = 125
                else:
                    canvas_center_x = 175 + self.detection_pan_x
                    canvas_center_y = 125 + self.detection_pan_y
            
            self.detection_canvas.create_image(canvas_center_x, canvas_center_y, 
                                             anchor=tk.CENTER, image=self.detection_photo)
            
            # Show zoom level if not 1.0
            if self.detection_zoom != 1.0:
                self.detection_canvas.create_text(10, 10, 
                                                text=f"Zoom: {int(self.detection_zoom * 100)}%",
                                                fill="yellow", anchor=tk.NW,
                                                font=('Segoe UI', 9, 'bold'))
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
        """Display a frame in the video canvas with timestamp"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((350, 250), Image.Resampling.LANCZOS)
        self.video_photo = ImageTk.PhotoImage(img)
        self.video_canvas.delete("all")
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.video_photo)
        
        # Display timestamp during playback
        if self.video_playing and self.video_cap:
            current_frame_idx = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # Subtract 1 because frame was just read
            fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                timestamp_sec = current_frame_idx / fps
                minutes = int(timestamp_sec // 60)
                seconds = timestamp_sec % 60
                timestamp = f"{minutes:02d}:{seconds:05.2f}"
                self.video_canvas.create_text(340, 240, text=timestamp, fill="yellow", anchor=tk.SE,
                                             font=('Segoe UI', 10, 'bold'))

    
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
    
    def display_current_frame(self, preserve_view: bool = False):
        """Display the current frame for annotation.
        preserve_view: when True, keep current zoom and pan across frames.
        """
        if not self.extracted_frames:
            return
        
        frame_number, frame = self.extracted_frames[self.current_frame_idx]
        
        # Convert frame to PhotoImage
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_frame_original = frame_rgb.copy()
        self.frame_height, self.frame_width = frame_rgb.shape[:2]
        
        img = Image.fromarray(frame_rgb)
        
        # Get actual canvas size (accounting for window/screen size)
        self.frame_canvas.update()
        actual_canvas_width = self.frame_canvas.winfo_width()
        actual_canvas_height = self.frame_canvas.winfo_height()
        
        # Use actual canvas size, with reasonable defaults
        canvas_width = max(actual_canvas_width, 100)
        canvas_height = max(actual_canvas_height, 100)
        
        # Calculate display size to fit canvas while maintaining aspect ratio
        # Keep display size consistent across frames of the same video to avoid bbox shifts
        if getattr(self, 'display_size_locked', False) and self.display_width_base and self.display_height_base:
            display_width = self.display_width_base
            display_height = self.display_height_base
        else:
            aspect_ratio = self.frame_width / self.frame_height
            canvas_aspect = canvas_width / canvas_height
            
            if aspect_ratio > canvas_aspect:
                # Image is wider - fit to width
                display_width = canvas_width
                display_height = int(canvas_width / aspect_ratio)
            else:
                # Image is taller - fit to height
                display_height = canvas_height
                display_width = int(canvas_height * aspect_ratio)
            # Cache and lock for this video
            self.display_width_base = display_width
            self.display_height_base = display_height
            self.display_size_locked = True
        
        self.display_width = display_width
        self.display_height = display_height

        # Base scale factors between original frame and fitted display image
        # These remain constant for a given video while display size is locked
        self.base_scale_x = self.display_width / float(self.frame_width)
        self.base_scale_y = self.display_height / float(self.frame_height)
        
        # Invalidate zoomed image cache when switching frames
        self.cached_zoomed_img = None
        self.cached_zoom_level = None
        
        # Load existing annotations AFTER display_width is set (base_scale available)
        self.load_existing_annotations()
        
        # Minimum zoom should be 1.0 (or less if needed to fit in canvas)
        # Calculate the zoom needed to fit image in canvas
        fit_zoom_width = canvas_width / self.display_width
        fit_zoom_height = canvas_height / self.display_height
        fit_zoom = min(fit_zoom_width, fit_zoom_height)
        
        # Minimum zoom is the fit zoom (allows zooming out to see full image)
        self.min_zoom = fit_zoom
        
        # Store original display image
        self.base_display_img = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
        
        if preserve_view:
            # Keep current zoom/pan, but ensure not below min
            if not hasattr(self, 'zoom_level'):
                self.zoom_level = self.min_zoom
            else:
                self.zoom_level = max(self.zoom_level, self.min_zoom)
            # keep pan as-is
        else:
            # Reset zoom and pan to fit image
            self.zoom_level = self.min_zoom
            self.center_image()
        
        # Apply zoom and pan
        self.update_zoomed_image()
        
        self.update_info_label()
    
    def load_existing_annotations(self):
        """Load existing annotations for the current frame.
        Stores boxes internally in ORIGINAL image pixel coordinates (x1,y1,x2,y2).
        """
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

                        # Convert normalized (relative to original frame) to ORIGINAL pixel coords
                        x_center_px = x_center * self.frame_width
                        y_center_px = y_center * self.frame_height
                        w_px = width * self.frame_width
                        h_px = height * self.frame_height

                        x1_orig = max(0.0, x_center_px - w_px / 2)
                        y1_orig = max(0.0, y_center_px - h_px / 2)
                        x2_orig = min(float(self.frame_width), x_center_px + w_px / 2)
                        y2_orig = min(float(self.frame_height), y_center_px + h_px / 2)

                        self.current_bboxes.append((class_id, int(x1_orig), int(y1_orig), int(x2_orig), int(y2_orig)))
    
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
    
    def center_image(self):
        """Center the image on the canvas"""
        self.frame_canvas.update()
        canvas_width = self.frame_canvas.winfo_width()
        canvas_height = self.frame_canvas.winfo_height()
        
        zoomed_width = int(self.display_width * self.zoom_level)
        zoomed_height = int(self.display_height * self.zoom_level)
        
        # Center the image
        self.pan_x = (canvas_width - zoomed_width) / 2
        self.pan_y = (canvas_height - zoomed_height) / 2
        
        # Ensure pan doesn't go too far
        self.pan_x = max(self.pan_x, 0)
        self.pan_y = max(self.pan_y, 0)
    
    def canvas_to_image_coords(self, canvas_x, canvas_y):
        """Convert canvas coordinates to ORIGINAL image coordinates accounting for zoom, pan, and base scale."""
        if not hasattr(self, 'base_scale_x') or not hasattr(self, 'base_scale_y'):
            return canvas_x, canvas_y
        # Remove pan and zoom, then unscale to original
        disp_x = (canvas_x - self.pan_x) / self.zoom_level
        disp_y = (canvas_y - self.pan_y) / self.zoom_level
        img_x = disp_x / self.base_scale_x
        img_y = disp_y / self.base_scale_y
        return img_x, img_y
    
    def image_to_canvas_coords(self, img_x, img_y):
        """Convert ORIGINAL image coordinates to canvas coordinates accounting for base scale, zoom and pan."""
        if not hasattr(self, 'base_scale_x') or not hasattr(self, 'base_scale_y'):
            return img_x, img_y
        disp_x = img_x * self.base_scale_x
        disp_y = img_y * self.base_scale_y
        canvas_x = disp_x * self.zoom_level + self.pan_x
        canvas_y = disp_y * self.zoom_level + self.pan_y
        return canvas_x, canvas_y
    
    def update_zoomed_image(self):
        """Update the displayed image with current zoom and pan (with caching for performance)"""
        if not hasattr(self, 'base_display_img'):
            return
        
        # Check if we can reuse cached zoomed image (only pan changed, not zoom)
        if (self.cached_zoomed_img is not None and 
            self.cached_zoom_level == self.zoom_level):
            # Just use cached zoomed image, avoid PIL resize
            self.frame_photo = ImageTk.PhotoImage(self.cached_zoomed_img)
        else:
            # Zoom changed, recalculate and cache the resized image
            zoomed_width = int(self.display_width * self.zoom_level)
            zoomed_height = int(self.display_height * self.zoom_level)
            
            # Resize image with zoom and cache for subsequent pans
            self.cached_zoomed_img = self.base_display_img.resize((zoomed_width, zoomed_height), Image.Resampling.LANCZOS)
            self.cached_zoom_level = self.zoom_level
            self.frame_photo = ImageTk.PhotoImage(self.cached_zoomed_img)
        
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
        min_allowed_zoom = self.min_zoom if hasattr(self, 'min_zoom') else 0.5
        if min_allowed_zoom <= new_zoom <= 10.0:
            self.zoom_level = new_zoom
            
            # Adjust pan to keep mouse position stable
            new_canvas_x, new_canvas_y = self.image_to_canvas_coords(img_x, img_y)
            self.pan_x += mouse_x - new_canvas_x
            self.pan_y += mouse_y - new_canvas_y
            
            self.update_zoomed_image()
    
    def reset_zoom(self):
        """Reset zoom and pan to fit image in canvas and center it"""
        self.zoom_level = self.min_zoom if hasattr(self, 'min_zoom') else 1.0
        self.center_image()
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
        
        # Constrain to ORIGINAL image boundaries
        img_x1 = max(0.0, min(img_x1, float(self.frame_width)))
        img_y1 = max(0.0, min(img_y1, float(self.frame_height)))
        img_x2 = max(0.0, min(img_x2, float(self.frame_width)))
        img_y2 = max(0.0, min(img_y2, float(self.frame_height)))
        
        # Ignore very small boxes (in image coordinates)
        if abs(img_x2 - img_x1) < 5 or abs(img_y2 - img_y1) < 5:
            if self.temp_rect:
                self.frame_canvas.delete(self.temp_rect)
                self.temp_rect = None
            return
        
        # Add bounding box in ORIGINAL image coordinates
        class_id = CLASS_MAPPING[self.current_class.get()]
        self.current_bboxes.append((class_id, int(img_x1), int(img_y1), int(img_x2), int(img_y2)))
        
        if self.temp_rect:
            self.frame_canvas.delete(self.temp_rect)
            self.temp_rect = None
        
        # Redraw all boxes
        self.update_zoomed_image()
    
    def on_right_click(self, event):
        """Handle right-click to delete bounding box"""
        # Convert canvas coordinates to image coordinates
        img_x, img_y = self.canvas_to_image_coords(event.x, event.y)
        
        # Check if click is inside any bounding box
        for i, (class_id, x1, y1, x2, y2) in enumerate(self.current_bboxes):
            if x1 <= img_x <= x2 and y1 <= img_y <= y2:
                # Delete this bounding box
                self.current_bboxes.pop(i)
                # Redraw all boxes
                self.update_zoomed_image()
                break
    
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
            self.display_current_frame(preserve_view=True)
    
    def next_frame(self, event):
        """Go to next frame"""
        self.save_current_frame_annotations()
        
        if self.current_frame_idx < len(self.extracted_frames) - 1:
            self.current_frame_idx += 1
            self.display_current_frame(preserve_view=True)
        else:
            self.update_status("📌 Last frame of this video. Press Enter to save and move to next video.", '#0066cc')
    
    def next_video(self):
        """Move to the next video"""
        self.save_current_frame_annotations()
        
        if self.current_video_idx < len(self.video_files) - 1:
            self.current_video_idx += 1
            self.load_current_video()
        else:
            self.update_status("🎉 All videos processed! Great work!", '#10b981')
    
    def prev_video(self):
        """Move to the previous video"""
        self.save_current_frame_annotations()
        
        if self.current_video_idx > 0:
            self.current_video_idx -= 1
            self.load_current_video()
            self.update_status("⬅️ Moved to previous video", '#0066cc')
        else:
            self.update_status("📌 Already at the first video", '#f59e0b')
    
    def on_detection_zoom(self, event):
        """Handle mouse wheel zoom on detection image"""
        # Determine zoom direction
        if event.num == 5 or event.delta < 0:
            # Zoom out
            zoom_factor = 0.9
        else:
            # Zoom in
            zoom_factor = 1.1
        
        # Update zoom level with constraints (0.5x to 3x)
        new_zoom = self.detection_zoom * zoom_factor
        if 0.5 <= new_zoom <= 3.0:
            self.detection_zoom = new_zoom
            self.display_detection_image()
    
    def reset_detection_zoom(self, event=None):
        """Reset detection image zoom to 1.0"""
        self.detection_zoom = 1.0
        self.detection_pan_x = 0
        self.detection_pan_y = 0
        self.display_detection_image()
    
    def on_detection_pan_start(self, event):
        """Start panning detection image with middle mouse button"""
        self.detection_panning = True
        self.detection_pan_start_x = event.x
        self.detection_pan_start_y = event.y
    
    def on_detection_pan_drag(self, event):
        """Pan the detection image"""
        if self.detection_panning:
            dx = event.x - self.detection_pan_start_x
            dy = event.y - self.detection_pan_start_y
            self.detection_pan_x += dx
            self.detection_pan_y += dy
            self.detection_pan_start_x = event.x
            self.detection_pan_start_y = event.y
            self.display_detection_image()
    
    def on_detection_pan_end(self, event):
        """End panning detection image"""
        self.detection_panning = False
    
    def on_enter_key(self, event):
        """Handle Enter key - save current video and move to next"""
        self.save_video_and_next()
    
    def save_video_and_next(self):
        """Save all annotations for current video and move to next video"""
        # First, save the current frame annotations
        self.save_current_frame_annotations()
        
        # Then save all frames of the current video that haven't been saved yet
        self.save_all_frames_of_current_video()
        
        # Save progress before moving to next video
        self.save_progress()
        
        # Move to next video
        if self.current_video_idx < len(self.video_files) - 1:
            self.current_video_idx += 1
            self.load_current_video()
            self.update_status("✅ Video saved! Moving to next video...", '#10b981')
        else:
            self.update_status("🎉 All videos have been processed! Excellent work!", '#10b981')
    
    def save_all_frames_of_current_video(self):
        """Save all extracted frames of the current video to output directory"""
        if not self.video_files or not self.extracted_frames:
            return

        basename = self.video_files[self.current_video_idx]
        output_path = self.output_dir.get()
        images_dir = os.path.join(output_path, "images")

        # Save exactly the frames currently extracted in memory
        for frame_number, frame in self.extracted_frames:
            frame_name = f"{basename}_{frame_number}.jpg"
            img_path = os.path.join(images_dir, frame_name)

            # Only save if doesn't exist (preserve already annotated frames)
            if not os.path.exists(img_path):
                cv2.imwrite(img_path, frame)
    
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
                    # Ensure coordinates are within original bounds
                    x1o = max(0, min(self.frame_width, x1))
                    y1o = max(0, min(self.frame_height, y1))
                    x2o = max(0, min(self.frame_width, x2))
                    y2o = max(0, min(self.frame_height, y2))

                    # Convert to normalized YOLO format relative to original frame size
                    x_center = ((x1o + x2o) / 2.0) / float(self.frame_width)
                    y_center = ((y1o + y2o) / 2.0) / float(self.frame_height)
                    width = abs(x2o - x1o) / float(self.frame_width)
                    height = abs(y2o - y1o) / float(self.frame_height)

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
        """Fast save and exit without reprocessing all videos"""
        # Save current frame and current video's pending frames (if any)
        self.update_status("💾 Saving current work and exiting...", '#10b981')
        self.save_current_frame_annotations()
        self.save_all_frames_of_current_video()
        self.save_progress()
        # Exit quickly (no blocking dialogs)
        self.master.after(100, self.master.quit)

################################################################################
# Main Entry Point
################################################################################

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoAnnotatorGUI(root)
    root.mainloop()
