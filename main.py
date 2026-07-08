from .model import GASNet
from .util import run_gasnet
import cv2
import torch
import numpy as np
from PIL import Image
import time


def generate_mask(image_pil, mask):
    return Image.fromarray((mask * 255).astype(np.uint8)).convert("L")

def remove_background(image_pil, mask):
    img_arr = np.array(image_pil.convert("RGB"))
    alpha = (mask * 255).astype(np.uint8)
    rgba = np.dstack([img_arr, alpha])
    return Image.fromarray(rgba, "RGBA")

def composite_on_background(rgba_image, bg_color=(0, 0, 0)):
    """Flatten RGBA onto solid colour — video containers don't carry alpha."""
    bg = Image.new("RGB", rgba_image.size, bg_color)
    bg.paste(rgba_image, mask=rgba_image.split()[3])
    return bg

def run_from_path(model, image_path, conf_threshold=0.55, n_runs=1):
    """
    Loads an image, runs it through run_gasnet, measures inference time/FPS, then
    displays results side by side or returns (mask, background-removed).
    """
    image_pil = Image.open(image_path).convert("RGB")

    # Warm-up (untimed) — avoids counting one-off CUDA init / kernel selection.
    _ = run_gasnet(model, image_pil, conf_threshold=conf_threshold)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(n_runs):
        mask, num_det = run_gasnet(model, image_pil, conf_threshold=conf_threshold)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_time = elapsed / n_runs
    fps = 1.0 / avg_time if avg_time > 0 else 0.0
    print(f"Inference time: {avg_time * 1000:.2f} ms  |  FPS: {fps:.2f}  "
          f"(avg over {n_runs} run{'s' if n_runs > 1 else ''})")

    mask_img = generate_mask(image_pil, mask).convert("RGB")
    removed_bg_img = composite_on_background(remove_background(image_pil, mask), bg_color=(0, 0, 0))

    return mask_img, removed_bg_img


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = "gasnet.pt" # point to gasnet model weights
    
    # load model
    gasnet_model = GASNet()
    gasnet_model.load_state_dict(torch.load(checkpoint, map_location=device))
    gasnet_model.to(device)
    gasnet_model.eval()
    
    mask, removed_bg = run_from_path(
        gasnet_model, 
        'image.jpg', # Replace `image.jpg` with actual image
        conf_threshold=0.7
    )



if __name__ == "__main__":
    main()
    
