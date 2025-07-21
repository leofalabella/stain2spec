from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from dataloaders.dataloader import HEAFDataModule
import torch
from models.firstmodel import AFNet, Pix2PixModel

wandb_logger = WandbLogger(project="stain2spec", name='he2ihc-baseline')

model = Pix2PixModel(lambda_L1=100.0, lambda_perceptual=10.0)
data = HEAFDataModule('data/processed/HE/train', 'data/processed/IHC/train')

trainer = Trainer(
    max_epochs=100, 
    logger=wandb_logger,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1,
    enable_progress_bar=True, 
    log_every_n_steps=50
)

trainer.fit(model, datamodule=data)