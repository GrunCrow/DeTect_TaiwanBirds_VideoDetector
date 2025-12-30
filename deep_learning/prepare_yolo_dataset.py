import argparse
import shutil
from pathlib import Path
import pandas as pd
import yaml


def resolve_path(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str)
    return (base_dir / p).resolve() if not p.is_absolute() else p.resolve()


def load_split(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if 'image_path' not in df.columns:
        raise ValueError(f"CSV {csv_path} must contain 'image_path'")
    if 'label_path' not in df.columns:
        raise ValueError(f"CSV {csv_path} must contain 'label_path'")
    return df


def copy_split(df: pd.DataFrame, base_dir: Path, out_images: Path, out_labels: Path):
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    copied, skipped = 0, 0
    for _, row in df.iterrows():
        img_src = resolve_path(row['image_path'], base_dir)
        lbl_src = resolve_path(row['label_path'], base_dir)
        if not img_src.exists():
            print(f"[WARN] Missing image: {img_src}")
            skipped += 1
            continue
        if not lbl_src.exists():
            print(f"[WARN] Missing label: {lbl_src}")
            skipped += 1
            continue
        shutil.copy2(img_src, out_images / img_src.name)
        shutil.copy2(lbl_src, out_labels / lbl_src.name)
        copied += 1
    print(f"Copied {copied} items into {out_images.parent.name}; skipped {skipped}")


def write_data_yaml(out_dir: Path, train_dir: Path, val_dir: Path, test_dir: Path, class_mapping_path: Path | None):
    data = {
        'path': str(out_dir.resolve()),
        'train': str(train_dir.resolve()),
        'val': str(val_dir.resolve()),
        'test': str(test_dir.resolve()),
    }
    if class_mapping_path and class_mapping_path.exists():
        # Expect CSV with id,name columns
        df = pd.read_csv(class_mapping_path)
        names = {}
        for _, row in df.iterrows():
            try:
                cid = int(row[0])
                cname = str(row[1])
                names[cid] = cname
            except Exception:
                continue
        data['names'] = names
    else:
        # fallback generic
        data['names'] = {0: 'object'}
    yaml_path = out_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(data, f)
    print(f"Wrote {yaml_path}")


def main():
    ap = argparse.ArgumentParser(description="Prepare YOLO dataset from CSV splits")
    ap.add_argument('--train-csv', required=True)
    ap.add_argument('--val-csv', required=True)
    ap.add_argument('--test-csv', required=True)
    ap.add_argument('--out-dir', default='runs/yolo_data')
    ap.add_argument('--base-dir', default='.', help='Base dir to resolve relative CSV paths')
    ap.add_argument('--classes-csv', default=None, help='Optional classes.csv (id,name)')
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    images_dir = out_dir / 'images'
    labels_dir = out_dir / 'labels'

    train_df = load_split(Path(args.train_csv))
    val_df = load_split(Path(args.val_csv))
    test_df = load_split(Path(args.test_csv))

    copy_split(train_df, base_dir, images_dir / 'train', labels_dir / 'train')
    copy_split(val_df, base_dir, images_dir / 'val', labels_dir / 'val')
    copy_split(test_df, base_dir, images_dir / 'test', labels_dir / 'test')

    write_data_yaml(
        out_dir=out_dir,
        train_dir=images_dir / 'train',
        val_dir=images_dir / 'val',
        test_dir=images_dir / 'test',
        class_mapping_path=Path(args.classes_csv) if args.classes_csv else None,
    )


if __name__ == '__main__':
    main()
