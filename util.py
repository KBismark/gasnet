
import numpy as np
import cv2
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Normalization constants 
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_input_tensor = torch.zeros(1, 3, 320, 320, dtype=torch.float32, device=DEVICE)

def preprocess(image_np: np.ndarray):
    """
    Optimized AMDS resize + reflection pad + normalization.
    Returns (tensor, pad_top, pad_left, new_h, new_w)
    """
    H, W = image_np.shape[:2]
    scale = 320 / max(H, W)
    new_w = int(W * scale)
    new_h = int(H * scale)

    image_r = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_h = 320 - new_h
    pad_w = 320 - new_w
    top   = pad_h // 2;  bottom = pad_h - top
    left  = pad_w // 2;  right  = pad_w - left

    image_p = cv2.copyMakeBorder(image_r, top, bottom, left, right, cv2.BORDER_REFLECT_101)

    # Normalized in-place to avoid a second allocation
    image_f = image_p.astype(np.float32) / 255.0
    image_f = (image_f - _MEAN) / _STD  

    # Copy into pre-allocated tensor (avoids torch.from_numpy + unsqueeze)
    _input_tensor[0].copy_(torch.from_numpy(image_f.transpose(2, 0, 1)))

    return _input_tensor, top, left, new_h, new_w


def postprocess(mask_u8: np.ndarray, min_blob_frac: float = 0.02) -> np.ndarray:
    """
    Operates on a 320x320 uint8 mask (values 0 or 255).
    Returns uint8 mask of same size.
    """
    if mask_u8.sum() == 0:
        return mask_u8

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )
    if num_labels <= 1:
        clean = mask_u8
    else:
        areas     = stats[1:, cv2.CC_STAT_AREA]
        total_fg  = areas.sum()
        threshold = max(1, int(total_fg * min_blob_frac))
        clean     = np.zeros_like(mask_u8)
        for label_idx, area in enumerate(areas, start=1):
            if area >= threshold:
                clean[labels == label_idx] = 255

    clean = cv2.dilate(clean, _MORPH_KERNEL)
    clean = cv2.erode(clean,  _MORPH_KERNEL)
    blurred = cv2.boxFilter(clean, ddepth=-1, ksize=(5, 5))
    _, clean = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    return clean


def run_gasnet(model, image_pil, conf_threshold=None):
    """
    Parameters
    ----------
    image_pil : PIL Image (any mode, converted to RGB internally)
    conf_threshold : float in [0,1] or None | Defaults to 0.55

    Returns
    -------
    (binary_mask np.uint8, num_detections=1) | binary_mask is 0/1 (not 0/255).     
    """
    threshold = conf_threshold if conf_threshold is not None else 0.55
    image_np = np.array(image_pil.convert("RGB"))
    H_orig, W_orig = image_np.shape[:2]

    tensor, top, left, new_h, new_w = preprocess(image_np)

    # Model inference
    with torch.inference_mode():
        output = model(tensor)
        # squeeze to (320,320), move to CPU once, stay as numpy
        pred   = output["mask"].squeeze().cpu().numpy()

    # Threshold - uint8 for OpenCV ops 
    mask_320 = ((pred > threshold) * 255).astype(np.uint8)
    
    # Postprocess at 320×320 before upsampling 
    mask_320 = _fast_postprocess(mask_320)

    # Remove padding and resize to original dimensions 
    mask_crop = mask_320[top:top + new_h, left:left + new_w]
    mask_orig = cv2.resize(mask_crop, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)

    # Return 0/1 binary mask (not 0/255)
    return (mask_orig > 0).astype(np.uint8), 1

