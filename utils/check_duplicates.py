"""
Scan data/PlantVillage for exact and near-duplicate images.
Outputs:
 - logs/duplicates_report.csv
 - data/processed/raw_cleaned/   (clean copy excluding duplicates)
 - data/removed_duplicates/      (moved duplicates if confirmed)
Usage:
    python utils/check_duplicates.py         # interactive (asks before moving duplicates)
    python utils/check_duplicates.py --auto  # auto-move duplicates
"""

import argparse
from pathlib import Path
import csv
from collections import defaultdict
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common_utils import (
    collect_images_by_class,
    md5_of_file,
    phash_of_image,
    copy_file,
    move_file,
    ensure_dir,
)
from tqdm import tqdm

ROOT = Path.cwd()
ORIGINAL_DIR = ROOT / "data" / "PlantVillage"
CLEANED_ROOT = ROOT / "data" / "processed" / "raw_cleaned"
REMOVED_DIR = ROOT / "data" / "removed_duplicates"
LOGS_DIR = ROOT / "logs"
REPORT_CSV = LOGS_DIR / "duplicates_report.csv"

PHASH_SIZE = 16
PHASH_HAMMING_THR = 6

def scan_and_report(auto_remove: bool = False):
    # Create necessary directories
    ensure_dir(CLEANED_ROOT)
    ensure_dir(REMOVED_DIR)
    ensure_dir(LOGS_DIR)

    print("\n" + "="*60)
    print("DUPLICATE DETECTION AND REMOVAL")
    print("="*60)

    # Check if source directory exists
    if not ORIGINAL_DIR.exists():
        print(f"ERROR: {ORIGINAL_DIR} not found!")
        print("Please ensure your data is in data/PlantVillage/")
        return

    images = collect_images_by_class(ORIGINAL_DIR)
    print(f"\n✓ Found {len(images)} images under {ORIGINAL_DIR}")
    
    # Get unique classes
    classes = set([c for _, c in images])
    print(f"✓ Found {len(classes)} classes")

    md5_map = dict()
    duplicates_exact = []
    phash_list = []

    print("\nStep 1: Computing MD5 hashes for exact duplicate detection...")
    for p, cls in tqdm(images, desc="MD5 hashing"):
        try:
            m = md5_of_file(p)
        except Exception as e:
            print(f"Failed MD5 for {p}: {e}")
            continue
        if m in md5_map:
            duplicates_exact.append((p, md5_map[m][0]))
        else:
            md5_map[m] = (p, cls)
            ph = phash_of_image(p, hash_size=PHASH_SIZE)
            phash_list.append((ph, p, m, cls))

    print(f"✓ Found {len(duplicates_exact)} exact duplicates")

    # Filter out None phashes
    phash_list_filtered = [x for x in phash_list if x[0] is not None]
    phash_list_sorted = sorted(phash_list_filtered, key=lambda t: int(str(t[0]), 16))
    duplicates_near = []

    print("\nStep 2: Computing perceptual hashes for near-duplicate detection...")
    n = len(phash_list_sorted)
    for i in tqdm(range(n), desc="Finding near-duplicates"):
        ph_i, p_i, md5_i, cls_i = phash_list_sorted[i]
        for j in range(i + 1, n):
            ph_j, p_j, md5_j, cls_j = phash_list_sorted[j]
            try:
                hamm = ph_i - ph_j
            except Exception:
                continue
            if hamm <= PHASH_HAMMING_THR:
                duplicates_near.append((p_j, p_i, hamm))

    print(f"✓ Found {len(duplicates_near)} near-duplicates")

    # Prepare removal map
    removed_map = dict()
    for dup, keeper in duplicates_exact:
        removed_map[str(dup.resolve())] = ("md5_exact", str(keeper.resolve()), None)
    for dup, keeper, hamm in duplicates_near:
        key = str(dup.resolve())
        if key not in removed_map:
            removed_map[key] = ("phash_near", str(keeper.resolve()), int(hamm))

    print(f"\n📊 Summary: {len(removed_map)} total duplicates to remove")

    # Write report
    ensure_dir(LOGS_DIR)
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["removed_path", "reason", "kept_path", "hamming_distance"])
        for removed_path, (reason, keeper, hamm) in removed_map.items():
            writer.writerow([removed_path, reason, keeper, hamm if hamm is not None else ""])
    print(f"✓ Report saved to: {REPORT_CSV}")

    # Create cleaned dataset
    print(f"\nStep 3: Creating cleaned dataset in {CLEANED_ROOT}")
    copied = 0
    for p, cls in tqdm(images, desc="Copying clean images"):
        if str(p.resolve()) in removed_map:
            continue
        dst = CLEANED_ROOT / cls / p.name
        copy_file(p, dst)
        copied += 1
    print(f"✓ Copied {copied} clean images")

    if not removed_map:
        print("\n✅ No duplicates detected! Dataset is clean.")
        return

    # Handle duplicate removal
    if auto_remove:
        confirm = "yes"
    else:
        print("\n" + "="*60)
        print("⚠️  CONFIRMATION REQUIRED")
        print(f"Found {len(removed_map)} duplicates")
        print("Move duplicates to data/removed_duplicates? (type 'yes' to confirm)")
        confirm = input("Your choice: ").strip().lower()

    if confirm == "yes":
        print(f"\nMoving {len(removed_map)} duplicate files...")
        for removed_path, (reason, keeper, hamm) in tqdm(removed_map.items(), desc="Moving duplicates"):
            src = Path(removed_path)
            if not src.exists():
                continue
            cls_name = src.parent.name if src.parent else "unknown"
            dst = REMOVED_DIR / cls_name / src.name
            try:
                move_file(src, dst)
            except Exception as e:
                print(f"Failed moving {src}: {e}")
        print("✓ Duplicates moved to:", REMOVED_DIR)
    else:
        print("⚠️  Duplicates not moved. Clean dataset still created in processed/raw_cleaned/")

    print("\n✅ Duplicate removal complete!")
    print(f"   Clean data: {CLEANED_ROOT}")
    print(f"   Report: {REPORT_CSV}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true", help="Automatically move duplicates without prompting")
    args = parser.parse_args()
    scan_and_report(auto_remove=args.auto)