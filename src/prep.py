"""Validate and deterministically split raw class folders."""
import argparse
import random
import shutil
from pathlib import Path
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", required=True)
    parser.add_argument("--train_out", required=True)
    parser.add_argument("--test_out", required=True)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--min_images", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    raw = Path(args.raw_data)
    class_dirs = sorted(p for p in raw.iterdir() if p.is_dir())
    if len(class_dirs) == 1:
        class_dirs = sorted(p for p in class_dirs[0].iterdir() if p.is_dir())
    if len(class_dirs) < 2:
        raise ValueError("Need at least two animal folders")
    rng = random.Random(args.seed)
    rows = []
    for class_dir in class_dirs:
        files = sorted(p for p in class_dir.iterdir() if p.is_file())
        for file in files:
            try:
                with Image.open(file) as image:
                    image.verify()
            except Exception as exc:
                raise ValueError(f"Unreadable image: {file} ({exc})") from exc
        if len(files) < args.min_images:
            raise ValueError(f"{class_dir.name} has fewer than {args.min_images} images")
        rng.shuffle(files)
        test_count = max(1, int(len(files) * args.test_ratio))
        test, train = files[:test_count], files[test_count:]
        for output, selected in ((Path(args.train_out), train), (Path(args.test_out), test)):
            destination = output / class_dir.name
            destination.mkdir(parents=True, exist_ok=True)
            for source in selected:
                shutil.copy2(source, destination / source.name)
        rows.append((class_dir.name, len(train), len(test)))
    print("Animal                 Train  Test")
    print("-----------------------------------")
    for name, train_count, test_count in rows:
        print(f"{name:<22} {train_count:>5} {test_count:>5}")


if __name__ == "__main__":
    main()