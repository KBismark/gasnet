

import cv2
import math
import numpy as np
from scipy.ndimage import distance_transform_edt

def amds_resize(image, mask, target_size=320):
    """
    Adaptive Maximum Dimension Scaling (AMDS)
    Preserves aspect ratio by scaling the largest image
    dimension to target_size.

    Returns resized image and mask.
    """

    h, w = image.shape[:2]

    scale = target_size / max(h, w)

    new_w = int(w * scale)
    new_h = int(h * scale)

    image = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    mask = cv2.resize(
        mask,
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST
    )

    return image, mask



def reflection_pad(image, mask, distance, boundary, target_size=320):
    h, w = image.shape[:2]
    pad_h = target_size - h
    pad_w = target_size - w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    # SDT padding represent "far outside", i.e. -1.
    distance = cv2.copyMakeBorder(
        distance, top, bottom, left, right, cv2.BORDER_CONSTANT, value=-1.0
    )
    # Boundary padding represent "background", i.e 0
    boundary = cv2.copyMakeBorder(
        boundary, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
    )
    return image, mask, distance, boundary


def create_signed_distance_map(mask):
    binary   = (mask > 0).astype(np.uint8)
    # Distance inside (positive)
    dist_in  = distance_transform_edt(binary)
    # Distance outside (negative)
    dist_out = distance_transform_edt(1 - binary)
    sdf      = dist_in - dist_out
    # Normalize to [-1, 1]
    max_val  = max(abs(sdf.max()), abs(sdf.min())) + 1e-6
    return (sdf / max_val).astype(np.float32)


def create_soft_boundary(signed_distance, sigma=0.05):
    """
    Generate a soft boundary target from the normalized
    Signed Distance Transform.

    signed_distance:
        normalized SDT in [-1,1]

    returns:
        boundary target in [0,1]
    """

    boundary = np.exp(
        -(signed_distance ** 2) /
        (2 * sigma ** 2)
    )

    return boundary.astype(np.float32)


def lr_lambda(epoch, warmup=warmup_epochs, total=total_epochs):
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))

