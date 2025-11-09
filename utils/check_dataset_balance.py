import os
from collections import defaultdict

base_dir = "data"
splits = ["train", "valid", "test"]

for split in splits:
    print(f"\n=== Checking {split} dataset ===")
    path = os.path.join(base_dir, split)
    if not os.path.exists(path):
        print("❌ Path not found:", path)
        continue

    counts = defaultdict(int)
    for cls in os.listdir(path):
        cls_path = os.path.join(path, cls)
        if os.path.isdir(cls_path):
            counts[cls] = len(os.listdir(cls_path))

    total = sum(counts.values())
    print(f"Total images in {split}: {total}")
    for cls, count in list(counts.items())[:5]:
        print(f"{cls}: {count} images")
    print("...")

print("\n✅ Done checking all splits.")
