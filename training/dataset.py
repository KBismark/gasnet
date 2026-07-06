from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .util import amds_resize, reflection_pad, create_signed_distance_map, create_soft_boundary


class SegDataset(Dataset):
    def __init__(self,root,split="train",transform=None,target_size=320):

        self.transform = transform
        self.target_size = target_size
        image_dir = Path(root) / split / "images"
        mask_dir = Path(root) / split / "masks"
        
        self.images = sorted(image_dir.glob("*.png"))
        self.masks = sorted(mask_dir.glob("*.png"))
        
        # ImageNet normalization (for pretrained MobileNetV3)
        self.mean = np.array([0.485, 0.456, 0.406],dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225],dtype=np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        # Load image
        image = cv2.imread(str(self.images[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        mask = cv2.imread(str(self.masks[idx]), cv2.IMREAD_GRAYSCALE)

        # Adaptive Maximum Dimension Scaling
        image, mask = amds_resize(image, mask, self.target_size)

        # Geometric supervision
        distance = create_signed_distance_map(mask)

        boundary = create_soft_boundary(distance)

        # Reflection padding
        image, mask, distance, boundary = reflection_pad(image,mask,distance,boundary,self.target_size)

        # Data augmentation
        if self.transform is not None:
            transformed = self.transform(
                image=image,
                mask=mask,
                distance=distance,
                boundary=boundary
            )

            image = transformed["image"]
            mask = transformed["mask"]
            distance = transformed["distance"]
            boundary = transformed["boundary"]

        # Binary mask
        mask = (mask > 127).astype(np.float32)

        # Binary boundary
        boundary = (boundary > 0.5).astype(np.float32)

        # Signed distance remains in [-1,1]
        distance = np.clip(distance, -1.0, 1.0).astype(np.float32)

        # Image normalization
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std

        # Convert to tensors
        image = torch.from_numpy(image.transpose(2, 0, 1) ).float()
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        distance = torch.from_numpy(distance).float().unsqueeze(0)
        boundary = torch.from_numpy(boundary).float().unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "distance": distance,
            "boundary": boundary
        }