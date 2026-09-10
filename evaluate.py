import time
import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import distance_transform_edt


def compute_boundary_metrics(pred_mask, gt_mask, theta=None, dilation_ratio=0.02):
    
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)

    # empty masks
    if not np.any(pred) and not np.any(gt):
        return {
            "boundary_f1": 1.0,
            "boundary_precision": 1.0,
            "boundary_recall": 1.0,
            "boundary_iou": 1.0,
        }
    if not np.any(pred) or not np.any(gt):
        return {
            "boundary_f1": 0.0,
            "boundary_precision": 0.0,
            "boundary_recall": 0.0,
            "boundary_iou": 0.0,
        }

    h, w = pred.shape
    if theta is None:
        diag = np.sqrt(h**2 + w**2)
        theta = max(1, int(round(dilation_ratio * diag)))  # Slack margin (approx 2-3px)

    # 1-pixel wide morphological boundary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pred_boundary = cv2.morphologyEx(pred, cv2.MORPH_GRADIENT, kernel) > 0
    gt_boundary = cv2.morphologyEx(gt, cv2.MORPH_GRADIENT, kernel) > 0

    if not np.any(pred_boundary) or not np.any(gt_boundary):
        return {
            "boundary_f1": 0.0,
            "boundary_precision": 0.0,
            "boundary_recall": 0.0,
            "boundary_iou": 0.0,
        }

    # Distance to the respective boundaries
    dist_to_gt = distance_transform_edt(~gt_boundary)
    dist_to_pred = distance_transform_edt(~pred_boundary)

    # Boundary Precision: Fraction of pred edges within theta distance from gt edges
    bp = np.sum(dist_to_gt[pred_boundary] <= theta) / (np.sum(pred_boundary) + 1e-7)

    # Boundary Recall: Fraction of gt edges within theta distance from pred edges
    br = np.sum(dist_to_pred[gt_boundary] <= theta) / (np.sum(gt_boundary) + 1e-7)

    bf1 = (2.0 * bp * br) / (bp + br) if (bp + br) > 0 else 0.0

    # Boundary IoU (mask dilation band intersection / union)
    d_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gt_band = np.logical_and(gt, cv2.dilate(gt_boundary.astype(np.uint8), d_kernel) > 0)
    pred_band = np.logical_and(pred, cv2.dilate(pred_boundary.astype(np.uint8), d_kernel) > 0)
    
    b_inter = np.logical_and(gt_band, pred_band).sum()
    b_union = np.logical_or(gt_band, pred_band).sum()
    boundary_iou = b_inter / b_union if b_union > 0 else 0.0

    return {
        "boundary_f1": float(bf1),
        "boundary_precision": float(bp),
        "boundary_recall": float(br),
        "boundary_iou": float(boundary_iou),
    }


def compute_metrics(pred_mask, gt_mask):
    """Computes full suite of standard and edge segmentation metrics."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    tn = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()

    union = tp + fp + fn
    iou = tp / union if union > 0 else 1.0

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    b_metrics = compute_boundary_metrics(pred_mask, gt_mask)

    return {
        "iou": float(iou),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        **b_metrics,
    }


def resize_mask_to_match(mask, target_shape):
    """Ensure predicted mask matches native ground-truth dimensions exactly."""
    if mask.shape == target_shape:
        return mask
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img = mask_img.resize((target_shape[1], target_shape[0]), Image.NEAREST)
    return (np.array(mask_img) > 127).astype(np.uint8)


def evaluate_pipeline_on_dataset(
    model,
    pipeline_fn,
    dataset,
    model_name="GASNet",
    dataset_name="PennFudan",
    conf_threshold=0.55,
    max_samples=None,
):
    records = []
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    is_cuda = torch.cuda.is_available()

    # WARM-UP
    print(f"Warming up for {model_name}...")
    warmup_n = min(5, n)
    for i in range(warmup_n):
        _ = pipeline_fn(dataset[i]["image"], model, conf_threshold)

    if is_cuda:
        torch.cuda.synchronize()
    print("Warm-up complete. Starting Evaluation...")

    # EVALUATION 
    for idx in range(n):
        sample = dataset[idx]
        image = sample["image"]
        gt_mask = sample["mask"]

        if is_cuda:
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        # INFERENCE
        pred_mask, num_detections = pipeline_fn(image, model, conf_threshold)

        if is_cuda:
            torch.cuda.synchronize()
        inference_time = time.perf_counter() - start_time

        pred_mask = resize_mask_to_match(pred_mask, gt_mask.shape)

        metrics = compute_metrics(pred_mask, gt_mask)

        records.append(
            {
                "model": model_name,
                "dataset": dataset_name,
                "filename": sample.get("filename", f"img_{idx}"),
                "num_detections": num_detections,
                "iou": metrics["iou"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "boundary_f1": metrics["boundary_f1"],
                "boundary_precision": metrics["boundary_precision"],
                "boundary_recall": metrics["boundary_recall"],
                "boundary_iou": metrics["boundary_iou"],
                "inference_time_sec": inference_time,
            }
        )

        if (idx + 1) % 50 == 0 or (idx + 1) == n:
            print(f"  [{model_name} / {dataset_name}] Processed {idx + 1}/{n} images")

    results_df = pd.DataFrame(records)

    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "num_images": n,
        "mean_iou": results_df["iou"].mean(),
        "mean_f1": results_df["f1"].mean(),
        "mean_boundary_f1": results_df["boundary_f1"].mean(),
        "mean_boundary_iou": results_df["boundary_iou"].mean(),
        "mean_accuracy": results_df["accuracy"].mean(),
        "mean_precision": results_df["precision"].mean(),
        "mean_recall": results_df["recall"].mean(),
        "mean_inference_time_sec": results_df["inference_time_sec"].mean(),
        "fps": 1.0 / results_df["inference_time_sec"].mean()
        if results_df["inference_time_sec"].mean() > 0
        else 0,
    }

    return results_df, summary

