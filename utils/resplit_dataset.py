import os
import shutil
import random
from tqdm import tqdm

random.seed(42)

# Paths
base_dir = "data"
old_train = os.path.join(base_dir, "train")
old_valid = os.path.join(base_dir, "valid")
new_train = os.path.join(base_dir, "train_new")
new_valid = os.path.join(base_dir, "valid_new")
new_test = os.path.join(base_dir, "test_new")

os.makedirs(new_train, exist_ok=True)
os.makedirs(new_valid, exist_ok=True)
os.makedirs(new_test, exist_ok=True)

# Combine old train + valid
all_classes = sorted(os.listdir(old_train))
for cls in tqdm(all_classes, desc="Processing classes"):
    train_cls_path = os.path.join(old_train, cls)
    valid_cls_path = os.path.join(old_valid, cls)

    all_imgs = []
    if os.path.exists(train_cls_path):
        all_imgs.extend([os.path.join(train_cls_path, x) for x in os.listdir(train_cls_path)])
    if os.path.exists(valid_cls_path):
        all_imgs.extend([os.path.join(valid_cls_path, x) for x in os.listdir(valid_cls_path)])

    random.shuffle(all_imgs)
    total = len(all_imgs)
    train_split = int(total * 0.8)
    valid_split = int(total * 0.9)

    # Prepare folders
    for d in [new_train, new_valid, new_test]:
        os.makedirs(os.path.join(d, cls), exist_ok=True)

    # Copy
    for i, img_path in enumerate(all_imgs):
        if i < train_split:
            dest = os.path.join(new_train, cls)
        elif i < valid_split:
            dest = os.path.join(new_valid, cls)
        else:
            dest = os.path.join(new_test, cls)
        shutil.copy(img_path, dest)

print("✅ Dataset successfully resplit into train_new / valid_new / test_new.")

