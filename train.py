import os
import cv2
import numpy as np
import torch
import albumentations as A
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from model import GASNet
from training.dataset import SegDataset
from training.loss_functions import total_loss
from training.util import lr_lambda

device = "cuda" if torch.cuda.is_available() else "cpu"
total_epochs = 50
BATCH_SIZE = 8


CHECKPOINT_DIR = "checkpoints"
LAST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "gasnet_last.pt")
BEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "gasnet_best.pt")


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.MotionBlur(p=0.2),
        ],
        additional_targets={
            "distance": "mask",
            "boundary": "mask",
        }
    )

    dataset_path = "data/COCO_person" 
    train_set = SegDataset(dataset_path, transform=train_transform, auto_split=True, split_ratio=0.80)
    val_set   = SegDataset(dataset_path, transform=train_transform, auto_split=True, split_ratio=0.20, is_val=True)

    num_workers = 2 if torch.cuda.is_available() else 0

    train_loader = DataLoader(
        train_set, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0)
    )

    model = GASNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))
    loss_fn = total_loss

    start_epoch = 0
    best_val_miou = 0.0

    if os.path.exists(LAST_CKPT_PATH):
        print(f"Found existing checkpoint at '{LAST_CKPT_PATH}'. Resuming training...")
        checkpoint = torch.load(LAST_CKPT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint and device == "cuda":
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_miou = checkpoint.get("best_val_miou", 0.0)
        print(f"Resumed successfully from Epoch {start_epoch} with Best Val mIoU: {best_val_miou:.4f}\n")
    else:
        print("No existing checkpoint found. Starting training from scratch...\n")

    for epoch in range(start_epoch, total_epochs):
        model.train()
        running = {}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs-1}")
        for batch in pbar:
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                outputs = model(batch["image"])
                loss, logs = loss_fn(outputs, batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + v.item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        n = len(train_loader)
        summary = " ".join(f"{k}:{v/n:.4f}" for k, v in running.items())
        print(f"\nEpoch {epoch} | {summary} | lr:{scheduler.get_last_lr()[0]:.6f}")

        # Validation 
        model.eval()
        val_loss_total = 0.0
        val_intersection = 0.0
        val_union = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
                with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                    outputs = model(batch["image"])
                    loss, _ = loss_fn(outputs, batch)
                
                val_loss_total += loss.item()

                pred_bin = (outputs["mask"] > 0.5).float()
                gt = batch["mask"]

                val_intersection += (pred_bin * gt).sum().item()
                val_union += (pred_bin + gt - pred_bin * gt).sum().item()

        val_miou = val_intersection / (val_union + 1e-6)
        avg_val_loss = val_loss_total / len(val_loader)
        print(f"Epoch {epoch} | Val Loss: {avg_val_loss:.4f} | Val mIoU: {val_miou:.4f}")

        ckpt_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_miou": best_val_miou,
            "val_loss": avg_val_loss
        }
        
        torch.save(ckpt_state, LAST_CKPT_PATH)

        # Save best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            ckpt_state["best_val_miou"] = best_val_miou
            torch.save(ckpt_state, BEST_CKPT_PATH)
            torch.save(model.state_dict(), "gasnet_best_weights.pt")
            print(f" -- New best mIoU: {best_val_miou:.4f}")


if __name__ == "__main__":
    main()
    
    