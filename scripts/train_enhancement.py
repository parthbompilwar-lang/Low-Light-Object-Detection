"""
Training script for Zero-DCE low-light enhancement model on ExDark dataset.
Zero-reference training using non-reference loss functions.
No OpenCV required (uses Pillow and PyTorch).
"""

import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from PIL import Image

from src.dataset import ExDarkDataset
from src.enhancement.model import ZeroDCE
from src.enhancement.loss import ZeroDCELoss

class PatchTransform:
    def __init__(self, size=(256, 256)):
        self.resize = transforms.Resize(size, antialias=True)
    def __call__(self, img_tensor):
        return self.resize(img_tensor)

def train_enhancement(
    epochs=10,
    batch_size=16,
    lr=1e-4,
    weight_decay=1e-4,
    target_exposure=0.6,
    weights_dir="weights"
):
    os.makedirs(weights_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Using device for training: {device}")
    if torch.cuda.is_available():
        print(f"--> GPU: {torch.cuda.get_device_name(0)}")

    # 1. Dataset & DataLoaders
    print("--> Initializing ExDark enhancement datasets...")
    transform = PatchTransform((256, 256))
    train_dataset = ExDarkDataset(root_dir=".", split="train", transform=transform, is_enhancement_only=True)
    val_dataset = ExDarkDataset(root_dir=".", split="val", transform=transform, is_enhancement_only=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    print(f"--> Train samples: {len(train_dataset)}, Batches per epoch: {len(train_loader)}")
    print(f"--> Val samples: {len(val_dataset)}, Batches: {len(val_loader)}")

    # 2. Model & Loss & Optimizer
    model = ZeroDCE().to(device)
    criterion = ZeroDCELoss(target_exposure=target_exposure).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    save_path = os.path.join(weights_dir, "enhancer_zero_dce.pth")

    print(f"\n--> Starting Zero-DCE training for {epochs} epochs...\n")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_spa = 0.0
        train_exp = 0.0
        train_col = 0.0
        train_tv = 0.0

        for batch_idx, img_batch in enumerate(train_loader):
            img_batch = img_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            enhanced, curve_params = model(img_batch)
            loss, loss_dict = criterion(img_batch, enhanced, curve_params)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss_dict["total"]
            train_spa += loss_dict["spa"]
            train_exp += loss_dict["exp"]
            train_col += loss_dict["col"]
            train_tv += loss_dict["tv"]

        num_train_batches = len(train_loader)
        avg_train_loss = train_loss / num_train_batches
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img_batch in val_loader:
                img_batch = img_batch.to(device, non_blocking=True)
                enhanced, curve_params = model(img_batch)
                _, loss_dict = criterion(img_batch, enhanced, curve_params)
                val_loss += loss_dict["total"]

        avg_val_loss = val_loss / len(val_loader)
        
        saved_flag = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
            }, save_path)
            saved_flag = " [* Best model saved]"

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {avg_train_loss:.4f} (Spa: {train_spa/num_train_batches:.4f}, Exp: {train_exp/num_train_batches:.4f}, Col: {train_col/num_train_batches:.4f}, TV: {train_tv/num_train_batches:.4f}) | "
              f"Val Loss: {avg_val_loss:.4f}{saved_flag}", flush=True)

    total_time = time.time() - start_time
    print(f"\n--> Training Complete in {total_time/60:.2f} minutes!", flush=True)
    print(f"--> Best weights saved to: {save_path} (Val Loss: {best_val_loss:.4f})", flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()
    train_enhancement(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
