"""
Configuration for the Label Review App
"""
from pathlib import Path
import os

# Dataset paths - MODIFY THIS TO YOUR ANNOTATED OUTPUT DIRECTORY
DATASET_PATH = Path(r'G:/2025-05-14_videos_annotated')
IMAGES_DIR = DATASET_PATH / 'images'
LABELS_DIR = DATASET_PATH / 'labels'

# Class mapping - matches your video annotation tool
CLASS_MAPPING = {
    0: 'Bat',
    1: 'Bird',
    2: 'Insect',
    3: 'Drone',
    4: 'Plane',
    5: 'Other',
    6: 'Unknown'
}

# Reverse mapping for quick lookup
CLASS_NAME_TO_ID = {v: k for k, v in CLASS_MAPPING.items()}

# Flask configuration
FLASK_DEBUG = True
HOST = '0.0.0.0'
PORT = 7700

# Image extensions
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

# Session configuration
CHANGES_FILE = 'changes.csv'
CHANGES_COLUMNS = ['video_path', 'frame_path', 'old_class', 'new_class', 'timestamp']
