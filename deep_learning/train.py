import argparse
from pathlib import Path
from ultralytics import YOLO
import random
import wandb
from PRIVATE import WANDB_API_KEY

def main():
    # ap = argparse.ArgumentParser(description="Train YOLOv11 with Ultralytics")
    # # ap.add_argument('--data', required=True, help='Path to data.yaml file')
    # ap.add_argument('--model', default='yolov11n.pt', help='YOLOv11 model checkpoint (e.g., yolov11n.pt)')
    # ap.add_argument('--epochs', type=int, default=100)
    # ap.add_argument('--imgsz', type=int, default=640)
    # ap.add_argument('--batch', type=int, default=16)
    # ap.add_argument('--workers', type=int, default=8)
    # ap.add_argument('--project', default='runs/train', help='Training output directory root')
    # ap.add_argument('--name', default='yolov11', help='Run name')
    # # ap.add_argument('--device', default=None, help='cuda device id or "cpu" (default: cpu)')
    # args = ap.parse_args()

    # data_path = Path(args.data).resolve()
    # if not data_path.exists():
    #     raise FileNotFoundError(f"data.yaml not found: {data_path}")
        
    # Initialize Weights & Biases environment
    wandb.login(key=WANDB_API_KEY)

    # Load a model
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

    # set cfg.yaml parameters
    # model.cfg.data = 'datasets/DeTect.yaml'  # path to data.yaml

    # Train the model
    results = model.train(
        data = 'cfg/datasets/DeTect.yaml',
        project = 'DeTect-BMMS',
        name = f'runs/yolov11n-test-cfg',
        epochs = 500,
        patience = 25,
        single_cls = True,
        # classes = [1],  # only these classes will be used for training - 1 = Bird
        # batch = 16,
        # imgsz = 640,
        # optimizer = 'auto',
        # deterministic = True,
        # single_cls = False,
        # cos_lr = False,
        # close_mosaic = 10,
        # dropout = 0.0,

        # Augmentation parameters - All false
        degrees = 0.0,
        translate = 0, # 0.1,
        scale = 0, #0.5,
        shear = 0.0,
        perspective = 0.0,
        flipud = 0.0,
        fliplr = 0, #0.5,
        mosaic = 0, #1.0,
        cutmix = 0.0,
        copy_paste = 0.0,
        # auto_augment = "randaugment",
        erasing = 0 # 0.4 # ---- change
        )

    print(results)


if __name__ == '__main__':
    main()
