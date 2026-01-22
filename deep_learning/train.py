import argparse
from pathlib import Path
from ultralytics import YOLO
import random
import wandb
from PRIVATE import WANDB_API_KEY

# EXP_NAME = "yolov11n-no_aug-DeTect500-v1"
EXP_NAME = "yolov26s-default-singlecls-bgundersampled-DeTect700-v1"

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
    model = YOLO("yolo26s.pt")  # load a pretrained model (recommended for training)

    # set cfg.yaml parameters
    # model.cfg.data = 'datasets/DeTect.yaml'  # path to data.yaml

    # Train the model
    results = model.train(
        data = 'cfg/datasets/DeTect.yaml',
        project = 'DeTect-BMMS',
        name = f'runs/{EXP_NAME}',
        device = 0, # use GPU 0
        epochs = 1500,
        patience = 250,
        single_cls = True,
        # classes = [1],  # only these classes will be used for training - 1 = Bird
        batch = 16,
        optimizer = "auto",
        # resume = False,
        # imgsz = 640,
        # optimizer = 'auto',
        # deterministic = True,
        # single_cls = False,
        # cos_lr = False,
        # close_mosaic = 10,
        # dropout = 0.0,

        # Hyperparameters
        # lr0 = 0.01,
        # lrf = 0.01,
        # momentum = 0.937,
        # weight_decay = 0.0005,
        # warmup_epochs = 3.0,
        # warmup_momentum = 0.8,
        # warmup_bias_lr = 0.1,
        # box = 7.5,
        # cls = 0.5,
        # dfl = 1.5,
        # nbs = 64,

        # Augmentation parameters - All false
        # hsv_h = 0.0, # 0.015
        # hsv_s = 0.0, # 0.7
        # hsv_v = 0.0, # 0.4
        # degrees = 0.0,
        # translate = 0, # 0.1,
        # scale = 0, #0.5,
        # shear = 0.0,
        # perspective = 0.0,
        # flipud = 0.0,
        # fliplr = 0, #0.5,
        # mosaic = 1.0, #1.0,
        # cutmix = 0.0,
        # copy_paste = 0.0,
        # # copy_paste_mode = 'flip', # mixup
        # # auto_augment = "randaugment",
        # erasing = 0 # 0.4 # ---- change
        )

    print(results)

    # ---------------------------------- Validation ----------------------------------

    print("\n------------------------------------------------------------------------\n")

    # run best model in val
    val_metrics = model.val(
        data = 'cfg/datasets/DeTect.yaml',
        split = 'val',
        project = 'DeTect-BMMS',
        name = f'runs/val/{EXP_NAME}',
        single_cls = True,
        save_json=True,
        plots=True
    )

    print("\n------------------------------------------------------------------------\n")
    # print("\nValidation metrics:\n")
    # print(val_metrics)

    print("\n------------------------------------------------------------------------\n")

    # ---------------------------------- Testing ----------------------------------
    test_metrics = model.val(
        data = 'cfg/datasets/DeTect.yaml',
        split = 'test',
        project = 'DeTect-BMMS',
        name = f'runs/test/{EXP_NAME}',
        single_cls = True,
        save_json=True,
        conf=0.1,
        plots=True
    )

    print("\n------------------------------------------------------------------------\n")
    # print("\nTest metrics:\n")
    # print(test_metrics)
    
    


if __name__ == '__main__':
    main()
