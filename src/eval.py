from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from dataloaders.dataloader import HEAFDataModule
import torch
from models.firstmodel import AFNet, Pix2PixModel
import matplotlib.pyplot as plt

def show(input_tensor, generated_tensor, target_tensor, n=3):
    """
    Show input, generated, and target images side by side.

    Parameters:
        input_tensor: (B,C,H,W) tensor of input images
        generated_tensor: (B,C,H,W) tensor of generated images
        target_tensor: (B,C,H,W) tensor of ground truth images
        n: how many images to show (default 3)
    """
    input_tensor = input_tensor.detach().cpu()
    generated_tensor = generated_tensor.detach().cpu()
    target_tensor = target_tensor.detach().cpu()
    
    for i in range(n):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Input image
        inp_img = input_tensor[i]
        inp_img = inp_img.permute(1, 2, 0)  # C,H,W -> H,W,C
        inp_img = inp_img.numpy()
        inp_img = (inp_img - inp_img.min()) / (inp_img.max() - inp_img.min())  # Normalize for display
        
        # Generated image
        gen_img = generated_tensor[i]
        gen_img = gen_img.permute(1, 2, 0).numpy()
        gen_img = (gen_img - gen_img.min()) / (gen_img.max() - gen_img.min())
        
        # Target image
        tgt_img = target_tensor[i]
        tgt_img = tgt_img.permute(1, 2, 0).numpy()
        tgt_img = (tgt_img - tgt_img.min()) / (tgt_img.max() - tgt_img.min())
        
        axes[0].imshow(inp_img)
        axes[0].set_title("Input")
        axes[0].axis('off')
        
        axes[1].imshow(gen_img)
        axes[1].set_title("Generated")
        axes[1].axis('off')
        
        axes[2].imshow(tgt_img)
        axes[2].set_title("Target")
        axes[2].axis('off')
        
        plt.show()

# Example usage:
model = Pix2PixModel()
data = HEAFDataModule('data/processed/HE/train', 'data/processed/IHC/train')
data.setup()

val_loader = data.val_dataloader()

# Suppose val_batch is a batch from your validation DataLoader
input_image, target_image = next(iter(val_loader))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
input_image = input_image.to(device)
target_image = target_image.to(device)

# Forward pass to generate images
with torch.no_grad():
    fake_image = model.generator(input_image)

show(input_image, fake_image, target_image, n=3)
