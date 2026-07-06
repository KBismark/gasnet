import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(weights="DEFAULT") # "DEFAULT"
        self.features = mobilenet.features
        # Stage indices for skip connection
        self.skip_idx = 3

    def forward(self, x):
        early_feat = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == self.skip_idx:
                early_feat = x    # save early feature map
        return x, early_feat  

class Head(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x):
        return self.net(x)


class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(2,8,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8,1,1),
            nn.Sigmoid()
        )

    def forward(self,feature,prior,boundary):
        prior = F.interpolate(prior,feature.shape[-2:],mode="bilinear",align_corners=False)
        boundary = F.interpolate(boundary,feature.shape[-2:],mode="bilinear",align_corners=False)
        prior_mag = torch.abs(prior)
        weight = self.attention(torch.cat([prior_mag, boundary], dim=1))

        return feature*weight


class Decoder(nn.Module):
    def __init__(self, in_ch, skip_ch):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch+skip_ch,128,3,padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128,64,3,padding=1),
            nn.ReLU(inplace=True)
        )

        self.out = nn.Conv2d(64,1,1)

    def forward(self, feature, skip, output_size):

        feature = F.interpolate(feature,scale_factor=4,mode="bilinear",align_corners=False)
        skip = F.interpolate(skip,feature.shape[-2:],mode="bilinear",align_corners=False)

        x = torch.cat([feature,skip],dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.out(x)
        x = F.interpolate(x,output_size,mode="bilinear",align_corners=False)
        
        return torch.sigmoid(x)


class GASNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.prior_head = Head(576)
        self.boundary_head = Head(576)
        self.fusion = Fusion()
        self.decoder = Decoder(576, 24)

    def forward(self,x):
        feature,skip = self.backbone(x)
        prior = torch.tanh(self.prior_head(feature))
        boundary = torch.sigmoid(self.boundary_head(feature))
        
        fused = self.fusion(feature,prior,boundary)
        mask = self.decoder(fused,skip,x.shape[-2:])

        prior_out = F.interpolate(prior,x.shape[-2:],mode="bilinear",align_corners=False)
        boundary_out = F.interpolate(boundary,x.shape[-2:],mode="bilinear",align_corners=False)

        return {
            "mask":mask,
            "prior":prior_out,
            "boundary":boundary_out
        }