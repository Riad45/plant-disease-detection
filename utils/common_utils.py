"""
Helper utilities for dataset preprocessing.
"""

from pathlib import Path
import hashlib
from PIL import Image
import imagehash
import shutil
from typing import List, Tuple
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS and p.is_file()

def collect_images_by_class(src_root: Path) -> List[Tuple[Path, str]]:
    """
    Returns list of (image_path, class_name) for all images under src_root.
    Assumes class directories are direct children of src_root.
    """
    out = []
    if not src_root.exists():
        return out
    for class_dir in sorted(src_root.iterdir()):
        if not class_dir.is_dir():
            continue
        for img in class_dir.rglob("*"):
            if is_image_file(img):
                out.append((img, class_dir.name))
    return out

def md5_of_file(path: Path, chunk_size: int = 8192) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def phash_of_image(path: Path, hash_size: int = 16):
    """
    Return perceptual hash (imagehash.ImageHash) or None if error.
    """
    try:
        img = Image.open(path)
        ph = imagehash.phash(img, hash_size=hash_size)
        img.close()
        return ph
    except Exception:
        return None

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def copy_file(src: Path, dst: Path):
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)

def move_file(src: Path, dst: Path):
    ensure_dir(dst.parent)
    shutil.move(str(src), str(dst))