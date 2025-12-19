import argparse
import os
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm", ".jfif"}

def list_images(folder: Path):
    files = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return files

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_files(files, dst_dir: Path):
    safe_mkdir(dst_dir)
    for src in files:
        # Keep filename; if collision, add suffix
        dst = dst_dir / src.name
        if dst.exists():
            dst = dst_dir / f"{src.stem}_{random.randint(100000,999999)}{src.suffix}"
        shutil.copy2(src, dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="raw")
    ap.add_argument("--out_dir", type=str, default="data")
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_per_class", type=int, default=0, help="0 means no limit")
    args = ap.parse_args()

    assert abs((args.train + args.val + args.test) - 1.0) < 1e-6, "train+val+test must sum to 1.0"

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir not found: {raw_dir.resolve()}")

    classes = [p.name for p in raw_dir.iterdir() if p.is_dir()]
    classes.sort()
    if not classes:
        raise RuntimeError(f"No class folders found in {raw_dir.resolve()}")

    random.seed(args.seed)

    # (Re)create output structure
    for split in ["train", "val", "test"]:
        for cls in classes:
            safe_mkdir(out_dir / split / cls)

    print(f"Found classes ({len(classes)}): {', '.join(classes)}")
    print(f"Splits: train={args.train}, val={args.val}, test={args.test}, seed={args.seed}")
    if args.max_per_class and args.max_per_class > 0:
        print(f"Max per class: {args.max_per_class}")

    total_copied = 0
    for cls in classes:
        cls_dir = raw_dir / cls
        imgs = list_images(cls_dir)
        if not imgs:
            print(f"[WARN] No images found in {cls_dir}")
            continue

        random.shuffle(imgs)
        if args.max_per_class and args.max_per_class > 0:
            imgs = imgs[: args.max_per_class]

        n = len(imgs)
        n_train = max(1, int(n * args.train))
        n_val = max(1, int(n * args.val))
        # ensure at least 1 test, and total == n
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
            # borrow from train first
            if n_train > 1:
                n_train -= 1
            else:
                n_val = max(1, n_val - 1)

        # final fix to match n
        while n_train + n_val + n_test > n:
            if n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                n_test = max(1, n_test - 1)

        while n_train + n_val + n_test < n:
            n_train += 1

        train_files = imgs[:n_train]
        val_files = imgs[n_train:n_train + n_val]
        test_files = imgs[n_train + n_val:]

        copy_files(train_files, out_dir / "train" / cls)
        copy_files(val_files, out_dir / "val" / cls)
        copy_files(test_files, out_dir / "test" / cls)

        total_copied += len(train_files) + len(val_files) + len(test_files)
        print(f"{cls}: total={n} -> train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    print(f"Done. Total files copied: {total_copied}")
    print(f"Output: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
