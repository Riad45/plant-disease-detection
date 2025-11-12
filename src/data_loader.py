"""
PyTorch DataLoader creation for train/val/test using torchvision.datasets.ImageFolder.
This loader accepts a 'model_type' argument to select image size and batch size from config.
"""

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# FIXED: Import from utils instead of src
try:
    from utils.config import config
except ImportError:
    # Alternative import for direct execution
    import sys
    sys.path.append('..')
    from utils.config import config

import torch
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transforms(img_size: int):
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.RandomResizedCrop((img_size, img_size), scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    val_test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return train_transforms, val_test_transforms

def get_dataloaders(model_type: str = "cnn"):
    """
    model_type: one of "cnn", "vit", "hybrid"
    Returns: train_loader, val_loader, test_loader, class_names
    """
    assert model_type in config.IMAGE_SIZES, f"Unknown model_type {model_type}"
    img_size = config.IMAGE_SIZES[model_type]
    batch_size = config.BATCH_SIZES[model_type]
    num_workers = config.NUM_WORKERS

    train_dir = Path(config.PROCESSED_DIR) / "train"
    val_dir = Path(config.PROCESSED_DIR) / "val"
    test_dir = Path(config.PROCESSED_DIR) / "test"

    if not train_dir.exists():
        print(f"ERROR: {train_dir} not found!")
        print("Please run the preprocessing pipeline first:")
        print("  1. python utils/check_duplicates.py")
        print("  2. python utils/split_data.py")
        return None, None, None, None

    train_t, val_t = get_transforms(img_size)
    train_dataset = ImageFolder(root=str(train_dir), transform=train_t)
    val_dataset = ImageFolder(root=str(val_dir), transform=val_t)
    test_dataset = ImageFolder(root=str(test_dir), transform=val_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)

    class_names = train_dataset.classes
    return train_loader, val_loader, test_loader, class_names

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING DATA LOADERS")
    print("="*60)
    
    # Test for each model type
    for model_type in ["cnn", "vit", "hybrid"]:
        print(f"\nTesting {model_type.upper()} configuration:")
        print("-"*40)
        
        train_loader, val_loader, test_loader, classes = get_dataloaders(model_type=model_type)
        
        if train_loader is None:
            print("Failed to load data!")
            break
            
        print(f"✓ Image size: {config.IMAGE_SIZES[model_type]}x{config.IMAGE_SIZES[model_type]}")
        print(f"✓ Batch size: {config.BATCH_SIZES[model_type]}")
        print(f"✓ Num classes: {len(classes)}")
        print(f"✓ Classes: {classes[:5]}..." if len(classes) > 5 else f"✓ Classes: {classes}")
        
        # Test loading a batch
        batch = next(iter(train_loader))
        imgs, labels = batch
        print(f"✓ Batch shape: {imgs.shape}")
        print(f"✓ Labels shape: {labels.shape}")
        print(f"✓ Data type: {imgs.dtype}")
        print(f"✓ Min value: {imgs.min():.3f}, Max value: {imgs.max():.3f}")
    
    print("\n✅ Data loader test complete!")