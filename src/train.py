from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from dataloaders.dataloader import HEAFDataModule, HFPairedDataModule, get_train_transforms
import torch
from models.firstmodel import AFNet, Pix2PixModel
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define transformations for data augmentation
train_transforms = get_train_transforms()
val_transforms = get_train_transforms()

wandb_logger = WandbLogger(project="stain2spec", name='he2ihc-baseline')

model = Pix2PixModel(lambda_L1=100.0, lambda_perceptual=100.0)
# data = HEAFDataModule('data/processed/HE/train', 'data/processed/IHC/train')
data = HFPairedDataModule(
    hf_id="wzhang472/HIT",
    batch_size=8,
    train_transform=train_transforms,
    val_transform=val_transforms
)

trainer = Trainer(
    max_epochs=300, 
    logger=wandb_logger,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1,
    enable_progress_bar=True, 
    log_every_n_steps=50,
    precision='16-mixed' # mixed precision fro faster training
)

trainer.fit(model, datamodule=data)