import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
import torchvision.utils as vutils
from torchvision.models import vgg16
from torchvision.models.feature_extraction import create_feature_extractor


class AFNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1)
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    

class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)] 
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        #encoder
        self.down1 = UNetDown(in_channels, 64, normalize=False) # No batchnorm on first layer
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)
        self.down6 = UNetDown(512, 512)
        self.down7 = UNetDown(512, 512)
        self.down8 = UNetDown(512, 512, normalize=False)  # bottleneck, no norm

        #decoder
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)  # 512 + 512 from skip
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        # Final layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], dim=1))
        u3 = self.up3(torch.cat([u2, d6], dim=1))
        u4 = self.up4(torch.cat([u3, d5], dim=1))
        u5 = self.up5(torch.cat([u4, d4], dim=1))
        u6 = self.up6(torch.cat([u5, d3], dim=1))
        u7 = self.up7(torch.cat([u6, d2], dim=1))
        out = self.final(torch.cat([u7, d1], dim=1))

        return out

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.model = nn.Sequential(
            # Input: (HE + IHC) → 6 channels
            nn.Conv2d(in_channels * 2, 64, 4, stride=2, padding=1),  # (B, 64, 128, 128)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),               # (B, 128, 64, 64)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),              # (B, 256, 32, 32)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, stride=2, padding=1),              # (B, 512, 16, 16)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),  # → (B, 512, 1, 1)
            nn.Flatten(),                 # → (B, 512)
            nn.Linear(512, 1)            # → (B, 1)
            # nn.Sigmoid()   #included in loss       # Optional: if you're using BCELoss
        )

    def forward(self, input_A, input_B):
        # concatenate HE and IHC images -> (B, 6, H, W)
        x = torch.cat((input_A, input_B), dim=1)
        return self.model(x)

# # test disciminator
# disc = Discriminator(3)
# dummy_input1 = torch.randn(1,3,256,256)
# dummy_input2 = torch.randn(1,3,256,256)
# out = disc(dummy_input1, dummy_input2)
# print(out.shape)

class Pix2PixModel(pl.LightningModule):
    def __init__(self, lambda_L1=10.0, lambda_perceptual=1.0):
        super().__init__()
        self.generator = GeneratorUNet()
        self.discriminator = Discriminator()
        self.automatic_optimization = False
        
        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

        self.generator.apply(init_weights)
        self.discriminator.apply(init_weights)
        # Losses
        self.adv_criterion = nn.BCEWithLogitsLoss()
        self.l1_criterion = nn.L1Loss()

        self.lambda_L1 = lambda_L1 
    
        # Load VGG for perceptual loss
        vgg = vgg16(pretrained=True).features.eval()
        for param in vgg.parameters():
            param.requires_grad = False # Freeze VGG

        # extract specific layers (ealy ones for texture)
        self.perceptual_extracor = create_feature_extractor(
            vgg, return_nodes={"3": "relu1_2"}
        )
        self.percetual_criterion = nn.L1Loss()
        self.lambda_perceptual = lambda_perceptual

    def training_step(self, batch, batch_idx):
        input_image, target_image = batch
        opt_d, opt_g = self.optimizers()

        # ---- Train dicriminator --------
        # Generate fake image
        fake_image = self.generator(input_image).detach() #dont update Gen

        # Disc on real pair
        real_pred = self.discriminator(input_image, target_image)
        real_loss = self.adv_criterion(real_pred, torch.ones_like(real_pred))

        # Disc on fake pair
        fake_pred = self.discriminator(input_image, fake_image)
        fake_loss = self.adv_criterion(fake_pred, torch.zeros_like(fake_pred))

        # combine
        d_loss = .5 * (real_loss +  fake_loss)

        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

        self.log("d_loss", d_loss, prog_bar=True)

        # ---- Train Generator --------
        # Gen fake image
        fake_image = self.generator(input_image) # Recompute fake (with grads)

        # Adversarial loss
        pred_fake = self.discriminator(input_image, fake_image)
        g_adv = self.adv_criterion(pred_fake, torch.ones_like(pred_fake))
        
        # L1 loss
        g_l1 = self.l1_criterion(fake_image, target_image)

        # Perceptual loss
        with torch.no_grad():
            target_feats = self.perceptual_extracor((target_image+1)/2)
        fake_feats = self.perceptual_extracor((fake_image + 1)/2)
        g_perc = self.percetual_criterion(fake_feats["relu1_2"], target_feats["relu1_2"])

        # combine
        g_loss = 1.0 * g_adv + self.lambda_L1 * g_l1 + self.lambda_perceptual * g_perc # trying to balance g_adv to stabalize training

        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        self.log("g_adv", g_adv, prog_bar=True)
        self.log("g_l1", g_l1, prog_bar=True)
        self.log("g_loss", g_loss, prog_bar=True)
        self.log("g_perceptual", g_perc, prog_bar=True)

    
    def validation_step(self, batch, batch_idx):
        input_image, target_image = batch

        # Generate image
        fake_image = self.generator(input_image)

        # Compute L1 loss (image similarity)
        val_l1 = self.l1_criterion(fake_image, target_image)

        # Log visuals once per epoch
        if batch_idx == 0:
            self.log_generated_images(input_image, target_image, fake_image, self.current_epoch)
        # Optionally: add PSNR/SSIM later

        self.log("val_l1", val_l1, prog_bar=True)

        return {"val_l1": val_l1}

        
    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
        return opt_d, opt_g
    
    def log_generated_images(self, he, af, fake_af, epoch):
        he = (he+1)/2
        af = (af+1)/2
        fake_af = (fake_af + 1)/2

        grid = vutils.make_grid(
            torch.cat([he[:4], af[:4], fake_af[:4]], dim=0),
            nrow=4, padding=2
        )

        self.logger.experiment.log({
            "Generated samples": [wandb.Image(grid, caption=f"Epoch {epoch}")],
            "epoch": epoch
        })