import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path

from model import GASNet
from training.dataset import SegDataset
from util import run_gasnet
from evaluate import evaluate_pipeline_on_dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


CHECKPOINT_PATH = "checkpoints/gasnet_best.pt"

model = GASNet().to(DEVICE)

print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

def run_gasnet_pipeline(image, model_obj, conf_threshold=0.55):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    pred_mask, num_det = run_gasnet(model_obj, image, conf_threshold=conf_threshold)
    return (pred_mask > 0).astype(np.uint8), num_det


pennfudan_dataset = SegDataset("data/PennFudanPed", image_dir="PNGImages", mask_dir="PedMasks", transform=None, auto_split=True, split_ratio=1.0, is_val=False )

eval_sets = {
    "PennFudan": pennfudan_dataset
}

all_results = []
all_summaries = []

for dataset_name, ds in eval_sets.items():
    print(f"\nEvaluating GASNet on {dataset_name} ({len(ds)} images)...")
    results_df, summary = evaluate_pipeline_on_dataset(
        model=model,
        pipeline_fn=run_gasnet_pipeline,
        dataset=ds,
        model_name="GASNet",
        dataset_name=dataset_name,
        conf_threshold=0.55,
    )
    all_summaries.append(summary)

    print(
        f"\n -- GASNet on {dataset_name}"
        f"\n    mIoU: {summary['mean_iou']:.4f} | Mask F1: {summary['mean_f1']:.4f}"
        f"\n    Boundary F1: {summary['mean_boundary_f1']:.4f} | Boundary IoU: {summary['mean_boundary_iou']:.4f}"
        f"\n    Precision: {summary['mean_precision']:.4f} | Recall: {summary['mean_recall']:.4f}"
        f"\n    FPS: {summary['fps']:.2f}"
    )

summary_df = pd.DataFrame(all_summaries)
summary_df.to_csv(f"{OUTPUT_DIR}/gasnet_summary.csv", index=False)

print("\n" + "=" * 80)
print(summary_df.to_string(index=False))
print("=" * 80)
print(f"Saved to '{OUTPUT_DIR}/'")

