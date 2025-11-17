from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import glob
import cv2 as cv
import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from PIL import Image
import datasets
from typing import Optional, Union, Sequence
import albumentations as A
from albumentations.pytorch import ToTensorV2

def hf_image_to_numpy(hf_image):
    """
    hf_image can be a dict like {'path': '...', 'bytes': b'...'} or a PIL.Image
    The huggingface image column is often a PIL.Image or dict with 'path' etc.
    This function tries common conversions to get HWC np.float32 array in [0,1].
    """
    if hf_image is None:
        raise ValueError("Found None hf_image")
    # If it's a PIL.Image
    if hasattr(hf_image, "convert"):
        img = hf_image.convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        return arr
    # If it's a dict with 'bytes' or 'path' or 'array'
    if isinstance(hf_image, dict):
        # Try 'bytes'
        if "bytes" in hf_image:
            img = Image.open(io.BytesIO(hf_image["bytes"])).convert("RGB")
            return np.array(img).astype(np.float32) / 255.0
        # Try 'path'
        if "path" in hf_image:
            img = Image.open(hf_image["path"]).convert("RGB")
            return np.array(img).astype(np.float32) / 255.0
        # Try 'array'
        if "array" in hf_image:
            arr = np.asarray(hf_image["array"]).astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            return arr
    # If it's already a numpy array
    if isinstance(hf_image, (np.ndarray,)):
        arr = hf_image.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return arr

    raise TypeError(f"Unhandled HF image type: {type(hf_image)}")

class HEAFDataset(Dataset):
    def __init__(self, he_dir, af_dir, transform=None):
        self.he_files = sorted([f for f in os.listdir(he_dir) if f.endswith('.npy')])
        self.he_dir = he_dir
        self.af_dir = af_dir
        self.transform = transform

    def __len__(self):
        return len(self.he_files)
    
    def __getitem__(self, idx):
        he = np.load(os.path.join(self.he_dir, self.he_files[idx])).astype(np.float32) # H,W,C
        af = np.load(os.path.join(self.af_dir, self.he_files[idx])).astype(np.float32) 

        if self.transform:
            transformed = self.transform(image=he, mask=af)
            he = transformed["image"]
            af = transformed["mask"]
        else:
            he = torch.tensor(np.transpose(he, (2, 0 ,1)), dtype=torch.float32) # C,H,W 
            af = torch.tensor(np.transpose(af, (2, 0 ,1)), dtype=torch.float32)

        return he, af
    
class HEAFDataModule(pl.LightningDataModule):
    def __init__(self, he_dir, af_dir, batch_size=8, train_transform=None):
        super().__init__()
        self.he_dir = he_dir
        self.af_dir = af_dir
        self.batch_size = batch_size
        self.train_transform = train_transform
        # self.val_transform = val_transform

    def setup(self, stage=None): 
        full_dataset = HEAFDataset(self.he_dir, self.af_dir, transform=self.train_transform)
        split = int(0.8 * len(full_dataset))
        self.train_dataset = torch.utils.data.Subset(full_dataset, list(range(split)))
        self.val_dataset = torch.utils.data.Subset(
            HEAFDataset(self.he_dir, self.af_dir, transform=A.Compose([
                A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5), max_pixel_value=1),
                ToTensorV2(transpose_mask=True)
            ])),
        list(range(split, len(full_dataset)))
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=15)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=15)

class PairedImageDataset(Dataset):
    def __init__(self, af_dir, he_dir, transform=None):
        self.af_paths = sorted(glob.glob(af_dir + '/*.png'))
        self.he_paths = sorted(glob.glob(he_dir + '/*.png'))
        self.transform = transform

    def __getitem__(self, idx):
        af = cv.imread(self.af_paths[idx])
        he = cv.imread(self.he_paths[idx])
        if self.transform:
            augmented = self.transform(image=af, mask=he)
            af, he = augmented['image'], augmented['mask']
        return af, he
    
    def __len__(self):
        return len(self.af_paths)

#-------------------- Hugging face dataloaders ----------
class HFDatasetPair(Dataset):
    def __init__(self,
                 ds_a: Union[Dataset, Sequence],
                 ds_b: Union[Dataset, Sequence],
                 transform: Optional[A.Compose] = None):
        # ds_a, ds_b are huggingface datasets.Dataset or simple sequences/lists of images
        self.ds_a = ds_a
        self.ds_b = ds_b
        self.transform = transform

        self._len = min(len(self.ds_a), len(self.ds_b)) # length is the min to avoid index errors

    def __len__(self):
        return self._len
    
    def __getitem__(self, idx):
        # get HF objects
        item_a = self.ds_a[idx]
        item_b = self.ds_b[idx]

        # Common column names: 'image', 'image_file', 'pixel_values' etc.
        # Try to be permissive:
        for key in ["image", "img", "file", "image_file", "pixel_values", "array"]:
            if isinstance(item_a, dict) and key in item_a:
                hf_img_a = item_a[key]
                break
        else:
            # maybe the dataset returns a PIL directly
            hf_img_a = item_a

        for key in ["image", "img", "file", "image_file", "pixel_values", "array"]:
            if isinstance(item_b, dict) and key in item_b:
                hf_img_b = item_b[key]
                break
        else:
            hf_img_b = item_b

        # convert to HWC numpy float32 in range [0,1]
        a_np = hf_image_to_numpy(hf_img_a)
        b_np = hf_image_to_numpy(hf_img_b)

        # Apply albumentations: note albumentations expects images in HWC [0..1] or [0..255]
        if self.transform:
            # use same API as before: transform(image=A, mask=B)
            transformed = self.transform(image=a_np, mask=b_np)
            a_t = transformed["image"]
            b_t = transformed["mask"]
        else:
            # Convert to tensors C,H,W
            a_t = torch.tensor(np.transpose(a_np, (2,0,1)), dtype=torch.float32)
            b_t = torch.tensor(np.transpose(b_np, (2,0,1)), dtype=torch.float32)

        return a_t, b_t
    
class HFPairedDataModule(pl.LightningDataModule):
    def __init__(self,
                 hf_id: str = "wzhang472/HIT",
                 split_a_name: str = "trainA",
                 split_b_name: str = "trainB",
                 batch_size: int = 8,
                 train_transform = None,
                 val_transform=None):
        super().__init__()
        self.hf_id = hf_id
        self.split_a_name = split_a_name
        self.split_b_name = split_b_name
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.val_transform = val_transform
        
    def setup(self, stage=None):
        # Clear HF cache if needed
        import gc
        gc.collect()
        # Load specific zip files directly (more memory efficient)
        print("Loading PAX5_trainA...")
        train_a = datasets.load_dataset(
            self.hf_id, 
            data_files="HIT/PAX5/PAX5_trainA.zip",
            split="train"
        )
        print("Loading PAX5_trainB...")
        train_b = datasets.load_dataset(
            self.hf_id,
            data_files="HIT/PAX5/PAX5_trainB.zip", 
            split="train"
        )
        print("Loading PAX5_testA...")
        test_a = datasets.load_dataset(
            self.hf_id,
            data_files="HIT/PAX5/PAX5_testA.zip",
            split="train"
        )
        print("Loading PAX5_testB...")
        test_b = datasets.load_dataset(
            self.hf_id,
            data_files="HIT/PAX5/PAX5_testB.zip",
            split="train"
        )

        print(f"Train A (H&E): {len(train_a)} images")
        print(f"Train B (IHC): {len(train_b)} images")
        print(f"Test A (H&E): {len(test_a)} images")
        print(f"Test B (IHC): {len(test_b)} images")

        # Build train dataset
        self.train_dataset = HFDatasetPair(train_a, train_b, transform=self.train_transform)
        self.val_dataset = HFDatasetPair(test_a, test_b, transform=self.val_transform)
        

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=False,
            persistent_workers=False
            )
            

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=False,
            persistent_workers=False
            )

## Data augmentation
def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.95, 1.05), translate_percent=(0.05, 0.05), rotate=(-15, 15), p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
        A.GaussianBlur(blur_limit=(3,5), p=0.2),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5,0.5,0.5), max_pixel_value=1), # assuming RGB
        ToTensorV2(transpose_mask=True)
    ])

# visualising data augmented
import matplotlib.pyplot as plt
import torchvision

def visualize_augmented_batch(x_batch, y_batch, n=4):
    """
    x_batch: HE images (input) tensor of shape [B, C, H, W]
    y_batch: AF/IHC images (target) tensor of shape [B, C, H, W]
    """
    x_batch = x_batch[:n].cpu()
    y_batch = y_batch[:n].cpu()
    print(f"x_batch shape {x_batch.shape}")
    print(f"y_batch shape {y_batch.shape}")

    for i in range(n):
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        for ax, img, title in zip(
            axs,
            [x_batch[i], y_batch[i]],
            ["HE (input)", "AF/IHC (target)"]
        ):
            # Unnormalize from [-1, 1] back to [0, 1] for display
            img = (img + 1) / 2.0
            # print(f"antes de permute {img.shape}")
            img = img.permute(1, 2, 0).numpy()
            # print(f"depois de permute {img.shape}")
            img = np.clip(img, 0, 1).astype(np.float32)
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.show()


#----------------- testing HF datasets ---------
if __name__ == "__main__":
    train_transform = get_train_transforms()
    val_transform = A.Compose([
        A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5), max_pixel_value=1),
        ToTensorV2(transpose_mask=True)
    ])

    dm = HFPairedDataModule(
        hf_id="wzhang472/HIT",
        split_a_name="PAX5_trainA",
        split_b_name="PAX5_trainB",
        batch_size=4,
        train_transform=train_transform,
        val_transform=val_transform
    )

    dm.setup()
    x, y = next(iter(dm.train_dataloader()))
    print(x.shape, y.shape)