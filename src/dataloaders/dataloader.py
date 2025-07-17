from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

class HEAFDataset(Dataset):
    def __init__(self, he_dir, af_dir):
        self.he_files = sorted([f for f in os.listdir(he_dir) if f.endswith('.npy')])
        self.he_dir = he_dir
        self.af_dir = af_dir

    def __len__(self):
        return len(self.he_files)
    
    def __getitem__(self, idx):
        he = np.load(os.path.join(self.he_dir, self.he_files[idx]))
        af = np.load(os.path.join(self.af_dir, self.he_files[idx]))

        he = torch.tensor(he.transpose(2,0,1), dtype=torch.float32) # HWC -> CHW
        af = torch.tensor(af, dtype=torch.float32).unsqueeze(0)     # Grayscale -> 1 channel

        return he, af
    
class HEAFDataModule(pl.LightningDataModule):
    def __init__(self, he_dir, af_dir, batch_size=8):
        super().__init__()
        self.he_dir = he_dir
        self.af_dir = af_dir
        self.batch_size = batch_size

    def setup (self, stage=None):
        full_dataset = HEAFDataset(self.he_dir, self.af_dir)
        split = int(0.8 * len(full_dataset))
        self.train_dataset = torch.utils.data.Subset(full_dataset, list(range(split)))
        self.val_dataset = torch.utils.data.Subset(full_dataset, list(range(split, len(full_dataset))))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)