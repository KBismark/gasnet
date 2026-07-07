import cv2
import numpy as np
import torch
import albumentations as A
from tqdm import tqdm
from .model import GASNet
from torch.utils.data import DataLoader
from training.dataset import SegDataset
from training.loss_functions import total_loss
from training.util import lr_lambda
from torch.optim.lr_scheduler import LambdaLR

device = "cuda" if torch.cuda.is_available() else "cpu"
total_epochs = 100


def main():
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

    # Load dataset
    dataset_path = "/data"
    train = SegDataset(dataset_path, split="train", transform=train_transform)
    val   = SegDataset(dataset_path, split="val")
    test  = SegDataset(dataset_path, split="test")

    train_loader = DataLoader(train, batch_size=8, shuffle=True)
    val_loader = DataLoader(val, batch_size=8, shuffle=False)

    # train
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    loss_fn = total_loss  
    
    model = GASNet().to(device) 

    for epoch in range(total_epochs):
        model.train()
        running = {}
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            outputs = model(batch["image"])
            loss, logs = loss_fn(outputs, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k, v in logs.items():
                running[k] = running.get(k, 0.0) + v.item()

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
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                outputs = model(batch["image"])
                loss, _ = loss_fn(outputs, batch)
                val_loss_total += loss.item()

                pred_bin = (outputs["mask"] > 0.5).float()
                gt = batch["mask"]

                val_intersection += (pred_bin * gt).sum().item()
                val_union += (pred_bin + gt - pred_bin * gt).sum().item()

        val_miou = val_intersection / (val_union + 1e-6)
        print(f"Epoch {epoch} | Val Loss: {val_loss_total/len(val_loader):.4f} | Val mIoU: {val_miou:.4f}")

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save(model.state_dict(), "gasnet_best.pt")
            print(f"  New best (Val mIoU {best_val_miou:.4f}), checkpoint saved.")
        

if __name__ == "__main__":
    main()