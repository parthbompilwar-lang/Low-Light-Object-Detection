"""
Training script for Faster R-CNN Object Detector on ExDark dataset.
Optimized for NVIDIA GeForce RTX 4050 GPU (CUDA) using PyTorch and Pillow.
Strictly avoids OpenCV.
"""

import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from src.dataset import ExDarkDataset, collate_fn, CLASS_NAMES
from src.detection.model import build_detector
from src.enhancement.model import ZeroDCE

def train_detector(
    epochs=5,
    batch_size=4,
    lr=0.0005,
    use_enhancer=True,
    enhancer_weights="weights/enhancer_zero_dce.pth",
    weights_dir="weights"
):
    os.makedirs(weights_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60, flush=True)
    print("       Faster R-CNN Detection Training on ExDark       ", flush=True)
    print("=" * 60, flush=True)
    print(f"--> Using device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"--> GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # 1. Load Enhancement Model if requested
    enhancer = None
    if use_enhancer:
        if os.path.exists(enhancer_weights):
            print(f"--> Loading Zero-DCE Enhancer from {enhancer_weights}...", flush=True)
            enhancer = ZeroDCE().to(device)
            ckpt = torch.load(enhancer_weights, map_location=device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            enhancer.load_state_dict(state_dict)
            enhancer.eval()
            for p in enhancer.parameters():
                p.requires_grad = False
            print("--> Zero-DCE Enhancer active: training detector on enhanced images!", flush=True)
        else:
            print(f"[!] Warning: Enhancer weights not found at {enhancer_weights}. Training on raw images.", flush=True)

    # 2. Datasets & Loaders
    print("--> Preparing ExDark train and val datasets...", flush=True)
    train_dataset = ExDarkDataset(root_dir=".", split="train", filter_empty=True)
    val_dataset = ExDarkDataset(root_dir=".", split="val", filter_empty=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    print(f"--> Train samples: {len(train_dataset)}, Batches: {len(train_loader)}", flush=True)
    print(f"--> Val samples: {len(val_dataset)}, Batches: {len(val_loader)}", flush=True)

    # 3. Model, Optimizer, Scheduler
    print("--> Initializing Faster R-CNN with ResNet-50-FPN backbone (mode='custom', 13 classes)...", flush=True)
    model = build_detector(num_classes=len(CLASS_NAMES), pretrained=True, mode="custom").to(device)

    # Fine-tuning parameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    best_val_loss = float("inf")
    save_path = os.path.join(weights_dir, "detector_fasterrcnn.pth")
    latest_path = os.path.join(weights_dir, "detector_fasterrcnn_latest.pth")

    print(f"\n--> Starting fine-tuning for {epochs} epochs with AMP mixed precision...\n", flush=True)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        cls_loss_total = 0.0
        box_loss_total = 0.0
        batch_count = 0

        for images, targets in train_loader:
            try:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

                # Enhance images through Zero-DCE if active
                if enhancer is not None:
                    with torch.no_grad():
                        enhanced_imgs = []
                        for img in images:
                            enh_img, _ = enhancer(img.unsqueeze(0))
                            enhanced_imgs.append(enh_img.squeeze(0))
                        images = enhanced_imgs

                optimizer.zero_grad()

                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        loss_dict = model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())
                    scaler.scale(losses).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    losses.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

                epoch_loss += losses.item()
                cls_loss_total += loss_dict.get("loss_classifier", torch.tensor(0.0)).item()
                box_loss_total += loss_dict.get("loss_box_reg", torch.tensor(0.0)).item()
                batch_count += 1

                if batch_count % 50 == 0 or batch_count == len(train_loader):
                    elapsed_b = time.time() - epoch_start
                    batches_left = len(train_loader) - batch_count
                    eta_sec = (elapsed_b / batch_count) * batches_left
                    print(f"  [Epoch {epoch:02d} | Batch {batch_count:>3}/{len(train_loader)}] Loss: {losses.item():.4f} (Cls: {loss_dict['loss_classifier'].item():.3f}, Box: {loss_dict['loss_box_reg'].item():.3f}) | ETA: {int(eta_sec//60)}m{int(eta_sec%60):02d}s", flush=True)

            except torch.OutOfMemoryError:
                print(f"[!] Warning: CUDA OOM at batch {batch_count}. Skipping batch and clearing cache.", flush=True)
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

        avg_train_loss = epoch_loss / batch_count
        scheduler.step()

        # Validation (Faster R-CNN returns losses in train mode)
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for val_images, val_targets in val_loader:
                try:
                    val_images = [img.to(device) for img in val_images]
                    val_targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in val_targets]

                    if enhancer is not None:
                        enhanced_val = []
                        for v_img in val_images:
                            e_img, _ = enhancer(v_img.unsqueeze(0))
                            enhanced_val.append(e_img.squeeze(0))
                        val_images = enhanced_val

                    if scaler is not None:
                        with torch.amp.autocast('cuda'):
                            v_loss_dict = model(val_images, val_targets)
                            v_losses = sum(loss for loss in v_loss_dict.values())
                    else:
                        v_loss_dict = model(val_images, val_targets)
                        v_losses = sum(loss for loss in v_loss_dict.values())

                    val_loss += v_losses.item()
                    val_batches += 1
                except torch.OutOfMemoryError:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

        avg_val_loss = val_loss / max(1, val_batches)

        # Always save latest
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val_loss,
            "classes": CLASS_NAMES,
        }, latest_path)

        saved_flag = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "classes": CLASS_NAMES,
            }, save_path)
            saved_flag = " [* Best detector saved]"

        epoch_time = (time.time() - epoch_start) / 60
        print(f"\n>>> Epoch [{epoch:02d}/{epochs:02d}] Finished in {epoch_time:.1f}m | Train Loss: {avg_train_loss:.4f} (Cls: {cls_loss_total/batch_count:.3f}, Box: {box_loss_total/batch_count:.3f}) | Val Loss: {avg_val_loss:.4f}{saved_flag}\n", flush=True)

    elapsed = (time.time() - start_time) / 60
    print(f"--> Detection training complete in {elapsed:.2f} minutes!", flush=True)
    print(f"--> Best weights saved to: {save_path} (Val Loss: {best_val_loss:.4f})", flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--use_enhancer", action="store_true", default=True, help="Use Zero-DCE enhanced images for training")
    args = parser.parse_args()

    train_detector(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_enhancer=args.use_enhancer
    )
