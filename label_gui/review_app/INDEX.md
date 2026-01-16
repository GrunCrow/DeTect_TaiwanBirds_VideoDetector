# 🎬 Label Review App - Start Here!

## Welcome! 👋

This is your complete **web-based video label review and relabeling application**. 

### What does it do?

✅ **View statistics** of your annotated dataset (frames, classes, coverage)  
✅ **Review videos** frame-by-frame with interactive visualization  
✅ **Relabel objects** by clicking on bounding boxes and changing their class  
✅ **Track all changes** and export them as CSV  
✅ **Share via network** - Access from other computers on your network  
✅ **Beautiful UI** - Modern, responsive, professional design  

---

## 🚀 Getting Started (2 Minutes)

### 1️⃣ Update Your Dataset Path

Open `config.py` with a text editor and change **line 5**:

```python
# CHANGE THIS TO YOUR ANNOTATED DATA PATH:
DATASET_PATH = Path(r'G:\2025-05-14_videos_annotated')
```

### 2️⃣ Install Dependencies

Open Command Prompt in this folder and run:
```bash
pip install -r requirements.txt
```

### 3️⃣ Launch the App

**Windows users:** Double-click `launch.bat`

**Or** open Command Prompt and run:
```bash
python app.py
```

### 4️⃣ Open in Browser

Go to: **http://localhost:5000**

Done! 🎉

---

## 📚 Documentation

| File | Purpose |
|------|---------|
| **QUICKSTART.md** | Quick setup guide (3 minutes) |
| **SETUP_GUIDE.txt** | Complete setup & features (comprehensive) |
| **README.md** | Full documentation, troubleshooting, API reference |
| **config.py** | ⚙️ Configuration - **YOU NEED TO EDIT THIS** |

---

## 🎯 Features at a Glance

### 📊 Dashboard (`http://localhost:5000/`)
```
┌─────────────────────────────────────────┐
│  Overview Statistics                    │
│  Total images: XXX | Annotated: XXX    │
│  Coverage: XX.X% | Targets: XXX        │
├─────────────────────────────────────────┤
│  Interactive Charts:                    │
│  • Class Distribution (bar)             │
│  • Videos per Class (bar)               │
│  • Annotated vs Background (pie)        │
│  • Frames per Video (stacked bar)       │
│  • Class Presence Heatmap               │
└─────────────────────────────────────────┘
```

### 🔍 Review & Relabel (`http://localhost:5000/review`)
```
┌──────────────┬─────────────────────────┐
│  Videos      │  Frame Display          │
│  • Video 1   │  [Image with Bboxes]    │
│  • Video 2   │  ◀ Prev | 1/50 | Next ▶ │
│  • Video 3   ├─────────────────────────┤
│              │  Bounding Boxes         │
│              │  ☑ BBox #1 - Bird       │
│              │  ☐ BBox #2 - Bat        │
│              ├─────────────────────────┤
│              │  Change Class           │
│              │  Current: Bird          │
│              │  New: [Select...]       │
│              │  [Confirm] [Cancel]     │
└──────────────┴─────────────────────────┘
```

---

## 🎬 How to Use - Step by Step

### Step 1: View Dataset Statistics
1. Open http://localhost:5000/
2. Scroll through the interactive charts
3. Understand your dataset composition

### Step 2: Find Videos to Review
1. Click "Review & Relabel" tab
2. See list of all videos with annotation counts
3. Click a video to load it

### Step 3: Browse Frames
1. Use "Previous" / "Next" buttons to navigate
2. Toggle "Show Bounding Boxes" to see/hide bboxes
3. Current frame number shows at bottom

### Step 4: Change a Label
1. **Click on a bbox** in the image OR in the "Bounding Boxes" list
2. The "Change Class" panel appears
3. Select new class from dropdown
4. Click "✅ Confirm Change"
5. Change is saved immediately!

### Step 5: Export Changes
1. When done, click "💾 Export Changes as CSV"
2. A file downloads: `label_changes_YYYYMMDD.csv`
3. Contains all changes with timestamps

---

## 🌐 Remote Access (For Team Collaboration)

Want others on your network to access the app?

**Step 1:** They're already allowed! (Host is set to `0.0.0.0` by default)

**Step 2:** Find your computer's IP address
```bash
# Windows: Open Command Prompt and type:
ipconfig

# Look for: IPv4 Address (e.g., 192.168.1.100)
```

**Step 3:** Share this URL with your team:
```
http://YOUR.IP.ADDRESS:5000
```

Example: `http://192.168.1.50:5000`

---

## 📋 File Structure

```
review_app/
│
├── 📄 app.py                 ← Main Flask app (don't edit unless you know Python)
├── ⚙️  config.py             ← **EDIT THIS** with your dataset path
├── 🛠️  utils.py              ← Helper functions (don't edit)
│
├── 📦 requirements.txt       ← Python packages (run pip install -r)
├── 🚀 launch.bat            ← Double-click to start (Windows)
│
├── 📚 README.md             ← Full documentation
├── ⚡ QUICKSTART.md         ← Quick setup (3 min)
├── 📖 SETUP_GUIDE.txt       ← Comprehensive guide
├── 🎯 INDEX.md              ← This file
│
├── templates/               ← HTML pages
│   ├── dashboard.html       ← Statistics page
│   └── review.html          ← Relabel page
│
└── static/                  ← Assets
    ├── css/style.css        ← Styling
    └── js/
        ├── dashboard.js     ← Dashboard logic
        └── review.js        ← Review logic
```

---

## ✅ Pre-Flight Checklist

Before launching, verify:

- [ ] `config.py` has your correct dataset path
- [ ] `images/` folder contains your frame images
- [ ] `labels/` folder contains matching `.txt` files
- [ ] Python is installed (`python --version`)
- [ ] You've run `pip install -r requirements.txt`
- [ ] Port 5000 is not blocked by firewall

---

## 🎨 Design Features

✨ **Modern UI**
- Gradient navbar
- Clean card-based layout
- Color-coded bounding boxes

📱 **Responsive**
- Works on desktop
- Mobile-friendly navigation
- Adapts to any screen size

⚡ **Fast**
- Canvas-based rendering
- Efficient image loading
- Instant feedback

🔒 **Safe**
- Changes saved immediately to files
- CSV export for audit trail
- No data loss

---

## 🆘 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| "Images not found" | Check `config.py` - wrong dataset path? |
| "Bboxes won't show" | Verify `.txt` files are in `labels/` folder |
| App crashes on startup | Check console for error messages |
| Can't access from another PC | Firewall? Use IP address instead of localhost |

See **README.md** for more troubleshooting.

---

## 📞 Need Help?

1. **Quick questions** → See `QUICKSTART.md`
2. **Setup issues** → See `SETUP_GUIDE.txt`
3. **How to use** → See `README.md`
4. **Code errors** → Check terminal/console for messages
5. **Browser errors** → Press F12, check "Console" tab

---

## 🎯 What's Next?

1. ✏️ Edit `config.py` with your dataset path
2. 📦 Install dependencies: `pip install -r requirements.txt`
3. 🚀 Launch: double-click `launch.bat` or run `python app.py`
4. 🌐 Open: http://localhost:5000
5. 📊 View dashboard
6. 🔍 Start reviewing and relabeling!

---

**Everything is ready to go!** 🎉

Your professional label review app is set up and waiting.

---

*DeTect Taiwan Birds - Label Review Tool*  
*v1.0 - January 2025*
