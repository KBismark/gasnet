

<h1 align="center"> Geometry-Aware Segmentation Network (GASNet)</h1>
<p align="center"><a href="https://huggingface.co/spaces/Kbis/segment-human">[Live Demo]</a></p>
 
<p align="center">
<img width="900" height="313" alt="Sample GASNet Segmented Image" src="https://github.com/user-attachments/assets/3984374f-1c65-4762-859f-02508eedd706" />
</p>

--- 

<p align="center">
GASNet is a lightweight single-class segmentation model built on MobileNetV3-Small. It predicts foreground masks using geometric priors (signed distance transform) and soft boundary maps. The network uses an attention-based fusion module to combine semantic features with spatial and boundary cues.
</p>

### Architecture

- **Backbone**: MobileNetV3-Small (ImageNet-pretrained)
- **Heads**: Dual-task heads produce spatial priors and boundary predictions
- **Fusion**: Attention module gates features using prior magnitude and boundary information
- **Decoder**: Upsamples fused features with skip connections to produce the final mask

### Loss Components

- **Mask loss**: Dice loss + boundary-weighted BCE
- **Spatial loss**: Smooth L1 between predicted prior and signed distance transform
- **Boundary loss**: BCE on soft boundary targets
- **Consistency loss**: Geometry consistency between mask and spatial prior

## Requirements

- torch
- torchvision
- Pillow
- numpy
- opencv-python
- scipy
- albumentations

## Dataset Structure

Organize your data as follows:

```
data/
├── train/
│   ├── images/
│   │   ├── img_001.png
│   │   ├── img_002.png
│   │   └── ...
│   └── masks/
│       ├── img_001.png
│       ├── img_002.png
│       └── ...
├── val/
│   ├── images/
│   │   └── ...
│   └── masks/
│       └── ...
└── test/
    ├── images/
    │   └── ...
    └── masks/
        └── ...
```

  



## Test Results   
NVIDIA T4 GPU | Colab | 16 RAM | Batch size = 8 | 50 Epochs     

Trained on the validation set of the COCO dataset due to resource limits. Only images containing at least one person were used for training. In all, there were 2693 images of the person class in the validation set. The encoder uses a MobileNetV3-Small backbone. Testing was done entirely on a different dataset using the Penn Fudan Pedestrian dataset. Results can be found in the summary CSV file in the results folder. A summary result is provided below:    

| Dataset | mIoU | Accuracy | Precision | Recall | FPS | Boundary F1 |
|---|---|---|---|---|---| --- |
| Penn-Fudan | 0.736 | 0.942 | 0.822 | 0.880 | 68.35 | 0.856 |


## Training

```bash
python train.py
```

Key hyperparameters:

- Target size: 320px (adaptive max-dimension scaling with reflection padding)
- Optimizer: AdamW (lr=3e-4)
- LR schedule: Cosine annealing with warmup

## Inference

```python
from model import GASNet
from util import run_gasnet
import torch
from PIL import Image

model = GASNet()
model.load_state_dict(torch.load("gasnet.pt", map_location="cpu"))
model.eval()

image = Image.open("image.jpg").convert("RGB")
mask, num_det = run_gasnet(model, image, conf_threshold=0.55)
```

`run_gasnet` returns a binary mask (0/1 uint8) resized to the original image dimensions.

