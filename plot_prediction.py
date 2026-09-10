import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset

from model import GASNet
from training.dataset import PennFudanDataset
from util import run_gasnet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_gasnet_pipeline(image, model_obj, conf_threshold=0.55):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # run_gasnet returns (0/1 mask, num_detections)
    pred_mask, num_det = run_gasnet(model_obj, image, conf_threshold=conf_threshold)
    return (pred_mask > 0).astype(np.uint8), num_det


def visualize_segmentation_comparison(model, pipeline_fn, samples, conf_threshold=0.55, save_path="results/gasnet_penn_fudan.png"):
    num_samples = len(samples)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, sample in enumerate(samples):
        img_pil = sample['image']
        gt_mask = sample['mask']
        name = sample.get('name', f"Sample {i+1}")

        # Run Inference
        pred_mask, num_det = pipeline_fn(img_pil, model, conf_threshold)

        # Plot Original Image
        axes[i, 0].imshow(img_pil)
        axes[i, 0].set_title(f"{name}\n(Detections: {num_det})", fontsize=12)
        axes[i, 0].axis('off')

        # Plot Ground Truth Mask
        axes[i, 1].imshow(gt_mask, cmap='gray')
        axes[i, 1].set_title("Ground Truth Mask", fontsize=12)
        axes[i, 1].axis('off')

        # Plot GASNet Predicted Mask
        axes[i, 2].imshow(pred_mask, cmap='gray') 
        axes[i, 2].set_title("GASNet Prediction", fontsize=12)
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Plot saved successfully to: {save_path}")
    plt.close(fig)


def main():
    CHECKPOINT_PATH = "checkpoints/gasnet_best.pt"
    if not os.path.exists(CHECKPOINT_PATH):
        CHECKPOINT_PATH = "gasnet_best.pt"

    model = GASNet().to(DEVICE)

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print("Model weights loaded successfully.")
    else:
        print(f"Warning: Checkpoint '{CHECKPOINT_PATH}' not found. Using untrained weights.")

    # Load dataset
    penn_dir = "data/PennFudanPed"
    pennfudan_dataset = PennFudanDataset(penn_dir)

    sample1 = pennfudan_dataset[0]  
    sample2 = pennfudan_dataset[57]  
    test_samples = [
        {'image': sample1['image'], 'mask': sample1['mask'], 'name': "PennFudan Sample 1"},
        {'image': sample2['image'], 'mask': sample2['mask'], 'name': "PennFudan Sample 2"}
    ]

    # Save comparison plot
    visualize_segmentation_comparison(
        model=model,
        pipeline_fn=run_gasnet_pipeline,
        samples=test_samples,
        conf_threshold=0.55,
        save_path=os.path.join(OUTPUT_DIR, "gasnet_penn_fudan.png")
    )


if __name__ == "__main__":
    main()
    
