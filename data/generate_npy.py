import numpy as np
import cv2
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

# Input and output directories
dirs = [
    "data/raw/BCI_dataset/BCI_dataset/HE/train",
    "data/raw/BCI_dataset/BCI_dataset/HE/test",
    "data/raw/BCI_dataset/BCI_dataset/IHC/train",
    "data/raw/BCI_dataset/BCI_dataset/IHC/test"
]
out_dirs = [
    "data/processed/HE/train",
    "data/processed/HE/test",
    "data/processed/IHC/train",
    "data/processed/IHC/test"
]

# Ensure output directories exist
for out_dir in out_dirs:
    os.makedirs(out_dir, exist_ok=True)

# Image processing function
def process_image(in_dir, out_dir, fname):
    if not fname.endswith('.png'):
        return
    img_path = os.path.join(in_dir, fname)
    img = cv2.imread(img_path)
    if img is None:
        return  # Skip unreadable or corrupted images
    img = cv2.resize(img, (256, 256))
    img = img / 255.0  # normalize
    out_path = os.path.join(out_dir, fname.replace('.png', '.npy'))
    np.save(out_path, img)

if __name__ == "__main__":
    for in_dir, out_dir in zip(dirs, out_dirs):
        fnames = os.listdir(in_dir)
        fnames = [f for f in fnames if f.endswith('.png')]  # only process .png files
        with ProcessPoolExecutor() as executor:
            # Use tqdm to wrap the iterator and show progress
            list(tqdm(
                executor.map(partial(process_image, in_dir, out_dir), fnames),
                total=len(fnames),
                desc=f"Processing {os.path.basename(out_dir)}"
            ))
