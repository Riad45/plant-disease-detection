"""
Check if any test images accidentally appear in training data by comparing filenames
"""
from pathlib import Path

def check_data_leakage_by_filename():
    print("🔍 CHECKING FOR DATA LEAKAGE BY FILENAME")
    print("=" * 60)
    
    train_dir = Path("data/processed/train")
    test_dir = Path("data/processed/test")
    val_dir = Path("data/processed/val")
    
    # Collect all files in each split
    train_files = {}
    test_files = {}
    val_files = {}
    
    for split_dir, files_dict in [(train_dir, train_files), (test_dir, test_files), (val_dir, val_files)]:
        if not split_dir.exists():
            print(f"❌ ERROR: {split_dir} not found!")
            return
        
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                images = [f.name for f in class_dir.glob("*") if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]
                files_dict[class_name] = set(images)
    
    # Check for leaks between train and test
    leaks_found = 0
    print("\n📊 Checking for duplicate files between splits:")
    
    for class_name in train_files:
        if class_name in test_files:
            overlap = train_files[class_name] & test_files[class_name]
            if overlap:
                print(f"❌ DATA LEAK in {class_name}: {len(overlap)} files in both train and test")
                leaks_found += len(overlap)
                for file in list(overlap)[:5]:  # Show first 5
                    print(f"   - {file}")
    
    for class_name in train_files:
        if class_name in val_files:
            overlap = train_files[class_name] & val_files[class_name]
            if overlap:
                print(f"❌ DATA LEAK in {class_name}: {len(overlap)} files in both train and val")
                leaks_found += len(overlap)
    
    if leaks_found == 0:
        print("✅ No filename-based data leakage detected")
    else:
        print(f"\n🚨 TOTAL LEAKS FOUND: {leaks_found}")
    
    return leaks_found

def check_class_distribution():
    """Check if any classes have very few images"""
    print("\n📈 CHECKING CLASS DISTRIBUTION")
    print("=" * 60)
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_dir = Path(f"data/processed/{split}")
        class_counts = {}
        
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                class_counts[class_dir.name] = len(images)
        
        print(f"\n{split.upper()} SET:")
        total_images = sum(class_counts.values())
        print(f"Total images: {total_images}")
        
        for cls, count in sorted(class_counts.items()):
            percentage = (count / total_images) * 100
            print(f"  {cls}: {count} images ({percentage:.1f}%)")
        
        # Check for very small classes
        small_classes = {cls: count for cls, count in class_counts.items() if count < 10}
        if small_classes:
            print(f"  ⚠️  SMALL CLASSES (may cause issues): {small_classes}")

if __name__ == "__main__":
    leaks = check_data_leakage_by_filename()
    check_class_distribution()
    
    if leaks > 0:
        print(f"\n🚨 CRITICAL: {leaks} data leaks found! Fix before training other models.")
    else:
        print(f"\n✅ Data looks clean! You can proceed with ViT training.")