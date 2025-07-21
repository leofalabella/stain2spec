import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from dataloaders.dataloader import HEAFDataModule, HEAFDataset
import torch
from models.firstmodel import AFNet, Pix2PixModel
from torch.utils.data import DataLoader, Dataset

wandb_logger = WandbLogger(project="stain2spec", name='he2af-baseline')

model = Pix2PixModel(lambda_L1=30.0, lambda_perceptual=5.0)
data = HEAFDataset('data/processed/HE/train', 'data/processed/IHC/train')

single_sample = [data[0]]

class SinglePairDataset(Dataset):
    def __init__(self, sample):
        self.sample = sample[0]

    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return self.sample

single_dataset = SinglePairDataset(single_sample)
single_loader = DataLoader(single_dataset, batch_size=1, shuffle=True)


class DebugDataModule(pl.LightningDataModule):
    def __init__(self, dataloader):
        super().__init__()
        self._loader = dataloader
    def train_dataloader(self):
        return self._loader
    def val_dataloader(self):
        return self._loader

debug_dm = DebugDataModule(single_loader)

trainer = Trainer(
    max_epochs=200, 
    logger=wandb_logger,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1,
    enable_progress_bar=True, 
    log_every_n_steps=50
)

trainer.fit(model, datamodule=debug_dm)