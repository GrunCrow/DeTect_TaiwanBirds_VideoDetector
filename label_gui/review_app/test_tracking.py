#!/usr/bin/env python
from pathlib import Path
import csv
from datetime import datetime
import sys

# Simulate creating a tracking entry
frame_image_path = r'G:\2025-05-14_videos_annotated\images\A03_021805cd-3457-38a7-a21c-8224fc84806f_79.jpg'
annotator = 'TestUser'

# Extract video name (new correct way)
frame_stem = Path(frame_image_path).stem
video_name = frame_stem.rsplit('_', 1)[0] if '_' in frame_stem else frame_stem
print(f'Extracted video_name: {video_name}')

# Create tracking directory and file
tracking_dir = Path('relabel_tracking')
tracking_dir.mkdir(exist_ok=True)

tracking_file = tracking_dir / f'relabel_log_test.csv'

# Write a tracking entry
row_data = {
    'timestamp': datetime.now().isoformat(),
    'annotator': annotator,
    'video': video_name,
    'frame': frame_stem,
    'old_class': 'Bird',
    'new_class': 'Plane',
    'image_path': frame_image_path
}

with open(tracking_file, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=row_data.keys())
    if f.tell() == 0:  # File is empty, write header
        writer.writeheader()
    writer.writerow(row_data)

print(f'Wrote tracking entry to {tracking_file}')

# Now simulate reading it back
print('\n--- Reading back from CSV ---')
with open(tracking_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f'Video: {row["video"]} | Annotator: {row["annotator"]} | {row["old_class"]} -> {row["new_class"]}')

# Clean up
import os
os.remove(tracking_file)
print(f'\nCleaned up test file')
