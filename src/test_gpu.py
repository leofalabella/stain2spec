import torch
import pytorch_lightning as pl
print(torch.__version__)


def check_pytorch_gpu():
    print("🔍 PyTorch GPU Check:")
    if torch.cuda.is_available():
        print(f"✅ GPU is available: {torch.cuda.get_device_name(0)}")
        print(f"    - Total GPUs: {torch.cuda.device_count()}")
        print(f"    - Current device: {torch.cuda.current_device()}")
    else:
        print("❌ No GPU available for PyTorch.")

def check_lightning_gpu():
    print("\n🔍 PyTorch Lightning Accelerator Check:")
    trainer = pl.Trainer(devices=1, accelerator='auto', enable_progress_bar=False)
    accelerator = trainer.accelerator
    print(f"✅ Lightning selected accelerator: {accelerator.__class__.__name__}")
    if accelerator.is_available:
        print("   -> CUDA is available for PyTorch Lightning.")
    else:
        print("   -> CUDA is NOT available for PyTorch Lightning.")

if __name__ == "__main__":
    check_pytorch_gpu()
    check_lightning_gpu()
