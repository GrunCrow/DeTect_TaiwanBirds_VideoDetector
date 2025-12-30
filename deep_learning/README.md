# YOLOv11 Training Pipeline

This folder provides a minimal pipeline to train YOLOv11 from CSV splits (`train.csv`, `val.csv`, `test.csv`). It prepares YOLO-format directories and launches training with Ultralytics.

## Expected CSV Schema

Each split CSV (`train.csv`, `val.csv`, `test.csv`) must include:
- `image_path`: path to the image (absolute or relative to repo root)
- `label_path`: path to the YOLO label file for that image (same basename, `.txt`)

Optional columns are ignored.

## Steps
1. **Install deps**
   ```bash
   pip install -r requirements.txt
   ```
2. **Prepare YOLO dataset structure**
   ```bash
   python prepare_yolo_dataset.py \
     --train-csv ../dataset/csvs/train.csv \
     --val-csv ../dataset/csvs/val.csv \
     --test-csv ../dataset/csvs/test.csv \
     --out-dir ./runs/yolo_data \
     --base-dir ..
   ```
   This creates:
   - `runs/yolo_data/images/{train,val,test}`
   - `runs/yolo_data/labels/{train,val,test}`
   - `runs/yolo_data/data.yaml`

3. **Train YOLOv11**
   ```bash
   python train_yolov11.py \
     --data ./runs/yolo_data/data.yaml \
     --model yolov11n.pt \
     --epochs 100 \
     --imgsz 640 \
     --batch 16 \
     --project ./runs/train \
     --name yolov11
   ```

4. **Validate** (optional)
   ```bash
   yolo val --data ./runs/yolo_data/data.yaml --weights ./runs/train/yolov11/weights/best.pt
   ```

## Notes
- If CSV paths are relative, set `--base-dir` to the repo root so files are resolved correctly.
- Labels must already be YOLO-format text files. No conversion of box coordinates is performed here—only copying and manifest creation.
- Adjust `--workers` and `--batch` to fit your hardware.
