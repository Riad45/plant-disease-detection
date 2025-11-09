import os
from collections import defaultdict

def count_images_in_dir(directory):
    """Count number of images in each class folder."""
    class_counts = defaultdict(int)
    total_images = 0

    if not os.path.exists(directory):
        print(f"⚠️  Directory not found: {directory}")
        return class_counts, total_images

    for class_name in sorted(os.listdir(directory)):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            count = len([
                f for f in os.listdir(class_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])
            class_counts[class_name] = count
            total_images += count

    return class_counts, total_images


def print_dataset_summary(name, directory):
    """Prints dataset class distribution summary."""
    print(f"\n=== Checking {name} dataset ===")
    class_counts, total = count_images_in_dir(directory)

    if total == 0:
        print(f"No images found in {directory}")
        return

    print(f"Total images in {name}: {total}\n")
    for class_name, count in class_counts.items():
        percent = (count / total) * 100
        print(f"{class_name:<35}: {count:>6} images ({percent:5.2f}%)")

    print("-" * 60)
    print(f"✅ {name} dataset check complete.\n")


if __name__ == "__main__":
    base_dir = "data"
    train_dir = os.path.join(base_dir, "train_new")
    valid_dir = os.path.join(base_dir, "valid_new")
    test_dir  = os.path.join(base_dir, "test_new")

    print_dataset_summary("train", train_dir)
    print_dataset_summary("valid", valid_dir)
    print_dataset_summary("test",  test_dir)
