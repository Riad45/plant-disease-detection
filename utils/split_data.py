"""
Create stratified train/val/test splits from data/processed/raw_cleaned.
Outputs:
 - data/processed/train, val, test
 - logs/split_summary.csv
 - data/processed/metadata.json
"""

import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common_utils import collect_images_by_class, copy_file, ensure_dir
from collections import Counter
import csv
from tqdm import tqdm

ROOT = Path.cwd()
CLEANED_ROOT = ROOT / "data" / "processed" / "raw_cleaned"
PROCESSED_ROOT = ROOT / "data" / "processed"
TRAIN_DIR = PROCESSED_ROOT / "train"
VAL_DIR = PROCESSED_ROOT / "val"
TEST_DIR = PROCESSED_ROOT / "test"
LOGS_DIR = ROOT / "logs"
SPLIT_CSV = LOGS_DIR / "split_summary.csv"
METADATA_JSON = PROCESSED_ROOT / "metadata.json"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
RANDOM_STATE = 42

def run_split():
    print("\n" + "="*60)
    print("DATA SPLITTING (80/10/10)")
    print("="*60)
    
    if not CLEANED_ROOT.exists():
        print(f"ERROR: {CLEANED_ROOT} not found!")
        print("Please run utils/check_duplicates.py first.")
        return
        
    ensure_dir(TRAIN_DIR)
    ensure_dir(VAL_DIR)
    ensure_dir(TEST_DIR)
    ensure_dir(LOGS_DIR)

    data = collect_images_by_class(CLEANED_ROOT)
    print(f"\n✓ Found {len(data)} images")
    
    paths = [str(p) for p, c in data]
    labels = [c for p, c in data]

    # Count classes
    label_counts = Counter(labels)
    print(f"✓ Found {len(label_counts)} classes")
    
    # Check for small classes
    small_classes = [lbl for lbl, cnt in label_counts.items() if cnt < 3]
    if small_classes:
        print(f"\n⚠️  WARNING: These classes have < 3 images: {small_classes}")
        print("Cannot perform stratified split with such small classes.")
        return

    print("\nStep 1: Creating train split (80%)...")
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=(1 - TRAIN_RATIO), stratify=labels, random_state=RANDOM_STATE
    )

    print("Step 2: Creating val/test splits (10% each)...")
    val_frac_of_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=(1 - val_frac_of_temp), stratify=temp_labels, random_state=RANDOM_STATE
    )

    def copy_list(paths_list, labels_list, dest_root):
        for p, lbl in tqdm(zip(paths_list, labels_list), total=len(paths_list)):
            src = Path(p)
            dst = Path(dest_root) / lbl / src.name
            copy_file(src, dst)

    print("\nStep 3: Copying files to split directories...")
    print("  Copying train files...")
    copy_list(train_paths, train_labels, TRAIN_DIR)
    print("  Copying val files...")
    copy_list(val_paths, val_labels, VAL_DIR)
    print("  Copying test files...")
    copy_list(test_paths, test_labels, TEST_DIR)

    # Write split summary
    print("\nStep 4: Generating reports...")
    with open(SPLIT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "class", "count"])
        for split_name, base in [("train", TRAIN_DIR), ("val", VAL_DIR), ("test", TEST_DIR)]:
            for cls_dir in sorted(base.iterdir()):
                if cls_dir.is_dir():
                    count = len(list(cls_dir.glob("*")))
                    writer.writerow([split_name, cls_dir.name, count])

    print(f"✓ Split summary saved to: {SPLIT_CSV}")

    # Write metadata
    metadata = {
        "created_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "split_ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "total_images": len(data),
        "num_classes": len(set(labels)),
        "random_state": RANDOM_STATE
    }
    with open(METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to: {METADATA_JSON}")

    # Print summary
    def class_counts(base):
        return sum(len(list(d.glob("*"))) for d in base.iterdir() if d.is_dir())

    print("\n" + "="*60)
    print("SPLIT SUMMARY")
    print("="*60)
    print(f"TRAIN: {class_counts(TRAIN_DIR)} images")
    print(f"VAL:   {class_counts(VAL_DIR)} images")
    print(f"TEST:  {class_counts(TEST_DIR)} images")
    print(f"TOTAL: {len(data)} images")
    print("\n✅ Data splitting complete!")

if __name__ == "__main__":
    run_split()