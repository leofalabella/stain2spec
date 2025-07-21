from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import glob
import cv2 as cv
import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

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
            A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5), max_pixel_value=1), ToTensorV2(transpose_mask=True)
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
        
## Data augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2
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


# # sanity check
# dm = HEAFDataModule("data/processed/HE/train", "data/processed/IHC/train", batch_size=4)
# dm.setup()
# x, y = next(iter(dm.train_dataloader()))
# print(x.shape, y.shape)

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

# dm = HEAFDataModule("data/processed/HE/train", "data/processed/IHC/train", batch_size=4, train_transform=get_train_transforms())
# dm.setup()
# x_batch, y_batch = next(iter(dm.train_dataloader()))
# visualize_augmented_batch(x_batch, y_batch, n=4)
# dm_without_transform = HEAFDataModule("data/processed/HE/train", "data/processed/IHC/train", batch_size=4)
# dm_without_transform.setup()
# x_batch, y_batch = next(iter(dm_without_transform.train_dataloader()))
# for i in range(len(x_batch)):
#     print(x_batch[i].shape)
