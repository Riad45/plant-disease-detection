"""
Verify no leakage between train/val/test splits.
Checks MD5 collisions across splits and filename collisions.
Writes logs/verification_report.txt
"""

from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common_utils import md5_of_file, is_image_file, ensure_dir
from collections import defaultdict
from tqdm import tqdm

ROOT = Path.cwd()
SPLIT_ROOT = ROOT / "data" / "processed"
LOGS_DIR = ROOT / "logs"
REPORT_FILE = LOGS_DIR / "verification_report.txt"

def gather_files(root: Path):
    out = []
    for split in ["train", "val", "test"]:
        sp = root / split
        if not sp.exists():
            continue
        for p in sp.rglob("*"):
            if is_image_file(p):
                out.append((split, p))
    return out

def run_verify():
    print("\n" + "="*60)
    print("DATA INTEGRITY VERIFICATION")
    print("="*60)
    
    ensure_dir(LOGS_DIR)
    
    files = gather_files(SPLIT_ROOT)
    print(f"\n✓ Found {len(files)} images across splits")

    md5_map = defaultdict(list)
    fname_map = defaultdict(list)

    print("\nStep 1: Computing MD5 hashes for all images...")
    for split, p in tqdm(files, desc="Hashing"):
        try:
            m = md5_of_file(p)
        except Exception as e:
            print(f"Error reading {p}: {e}")
            continue
        md5_map[m].append((split, p))
        fname_map[p.name].append((split, p))

    # Check for leakage
    leaked = {m: lst for m, lst in md5_map.items() if len({s for s, _ in lst}) > 1}
    fname_collisions = {fn: lst for fn, lst in fname_map.items() if len({s for s, _ in lst}) > 1}

    print("\nStep 2: Checking for data leakage...")
    
    # Write report
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("DATA INTEGRITY VERIFICATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total images scanned: {len(files)}\n")
        f.write(f"Train images: {sum(1 for s, _ in files if s == 'train')}\n")
        f.write(f"Val images: {sum(1 for s, _ in files if s == 'val')}\n")
        f.write(f"Test images: {sum(1 for s, _ in files if s == 'test')}\n\n")

        if leaked:
            f.write("⚠️  DATA LEAKAGE DETECTED!\n")
            f.write("The following images appear in multiple splits:\n\n")
            for m, lst in leaked.items():
                f.write(f"MD5: {m}\n")
                for s, p in lst:
                    f.write(f"  {s}: {p}\n")
            f.write("\n")
            print(f"⚠️  WARNING: Found {len(leaked)} MD5 collisions across splits!")
        else:
            f.write("✓ No MD5 leakage found across splits.\n\n")
            print("✓ No MD5 leakage detected")

        if fname_collisions:
            f.write("⚠️  FILENAME COLLISIONS:\n")
            f.write("Same filenames in different splits (different images):\n\n")
            for fn, lst in fname_collisions.items():
                f.write(f"Filename: {fn}\n")
                for s, p in lst:
                    f.write(f"  {s}: {p}\n")
            f.write("\n")
            print(f"⚠️  WARNING: Found {len(fname_collisions)} filename collisions")
        else:
            f.write("✓ No filename collisions across splits.\n\n")
            print("✓ No filename collisions")

    print(f"\n✓ Report saved to: {REPORT_FILE}")
    
    if leaked:
        print("\n⚠️  CRITICAL: Data leakage detected! Review the report.")
    else:
        print("\n✅ Data integrity verified - No leakage found!")

if __name__ == "__main__":
    run_verify()