# Quick Start Guide

## Setup (One-time)

### 1. Update Dataset Path
Open `config.py` and change line 5:
```python
DATASET_PATH = Path(r'G:\2025-05-14_videos_annotated')  # YOUR PATH HERE
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the App

### Option A: Windows (Easy)
Double-click `launch.bat`

### Option B: Command Line
```bash
python app.py
```

The app will start on **http://localhost:5000**

---

## Remote Access

To allow other computers on your network to access the app:

1. Edit `config.py`:
   ```python
   HOST = '0.0.0.0'  # Already set by default
   ```

2. Find your computer's IP address:
   - Open Command Prompt
   - Type: `ipconfig`
   - Look for "IPv4 Address" (something like 192.168.x.x)

3. Other computers can access it at:
   ```
   http://YOUR.IP.ADDRESS:5000
   ```

Example: `http://192.168.1.100:5000`

---

## Using the App

### Dashboard (Statistics)
- View overall dataset statistics
- Explore interactive charts
- Analyze class distribution and frame coverage

### Review & Relabel
1. Click on a video to load its frames
2. Use Previous/Next to browse frames
3. Click on a bounding box to select it
4. Choose a new class from the dropdown
5. Click "Confirm Change" to update the label
6. Export changes as CSV when done

---

## Need Help?

See `README.md` for:
- Detailed feature documentation
- Troubleshooting guide
- Architecture overview
- API reference
