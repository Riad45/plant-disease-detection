import os
import torch
from datetime import datetime
from pathlib import Path

class Config:
    # Root
    ROOT_DIR = Path(__file__).parent.parent
    DATA_DIR = ROOT_DIR / "data"
    PROCESSED_DIR = DATA_DIR / "processed"
    LOG_DIR = ROOT_DIR / "logs"
    SAVED_MODELS_DIR = ROOT_DIR / "saved_models"
    
    # Create directories if they don't exist
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Print GPU info
    if torch.cuda.is_available():
        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"Using GPU: {GPU_NAME} ({GPU_MEMORY:.1f} GB)")
    else:
        print("Using CPU")

    # Per-model image sizes (px, square)
    IMAGE_SIZES = {
        "cnn": 300,     # EfficientNet-B3 native
        "vit": 224,     # ViT default
        "hybrid": 256   # compromise for hybrid
    }

    # OPTIMIZED BATCH SIZES for RTX 3050 6GB
    # With Mixed Precision Training (FP16)
    BATCH_SIZES_FP16 = {
        "cnn": 24,      # Can handle 24 with mixed precision
        "vit": 16,      # ViT can handle 16 with mixed precision
        "hybrid": 12    # Hybrid needs more memory
    }
    
    # Without Mixed Precision Training (FP32)
    BATCH_SIZES_FP32 = {
        "cnn": 16,      # Max 16 without mixed precision
        "vit": 10,      # ViT needs more memory
        "hybrid": 8     # Hybrid is most memory intensive
    }
    
    # Auto-select based on available memory and mixed precision support
    USE_MIXED_PRECISION = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
    BATCH_SIZES = BATCH_SIZES_FP16 if USE_MIXED_PRECISION else BATCH_SIZES_FP32

    # DataLoader workers
    NUM_WORKERS = 2

    # Training defaults
    NUM_EPOCHS = 50
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING_PATIENCE = 10
    
    # Number of classes (Will be set dynamically after data loading)
    NUM_CLASSES = None

    # Misc
    RANDOM_SEED = 42
    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_NAME = f"experiment_{RUN_ID}"
    
    # Method to update num_classes
    def update_num_classes(self, num_classes):
        self.NUM_CLASSES = num_classes
        print(f"Updated NUM_CLASSES to: {num_classes}")

config = Config()