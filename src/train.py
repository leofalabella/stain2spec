from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from dataloaders.dataloader import HEAFDataModule
import torch
from models.firstmodel import AFNet

wandb_logger = WandbLogger(project="stain2spec", name='he2af-baseline')

model = AFNet()
data = HEAFDataModule('data/processed/HE_resized', 'data/processed/AF_resized')

trainer = Trainer(
    max_epochs=20, 
    logger=wandb_logger,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu'
)

trainer.fit(model, datamodule=data)