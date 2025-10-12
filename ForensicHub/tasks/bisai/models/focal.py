# ------------------------------------------------------------------------------
# FOCAL backbone integration for ForensicHub
# Author: Auto-generated (modified for fixed 512x512 output)
# ------------------------------------------------------------------------------

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from ForensicHub.registry import register_model

# ---------------------------------------------------------
# Local/Detect heads
# ---------------------------------------------------------
class UpsampleLocalHead(nn.Module):
    """Local mask prediction head with fixed output size 512x512"""
    def __init__(self, in_channels, out_channels=1, output_size=(512, 512)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.relu = nn.ReLU(inplace=True)
        self.out_conv = nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)
        self.output_size = output_size  # 固定输出尺寸

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.out_conv(x)
        # 固定输出为 512x512
        x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        return x

class DetectHead(nn.Module):
    """Global label prediction head"""
    def __init__(self, in_channels, hidden_dim=256, out_dim=1):
        super().__init__()
        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        self.pool_max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        avg = self.pool_avg(x).flatten(1)
        max_ = self.pool_max(x).flatten(1)
        feat = torch.cat([avg, max_], dim=1)
        return self.mlp(feat)

# ---------------------------------------------------------
# Import FOCAL HRNet backbone
# ---------------------------------------------------------
sys.path.append('/data/disk2/yer/ForensicHub/external/FOCAL/models/')
from hrnet import FOCAL_HRNet

# ---------------------------------------------------------
# Focal Model
# ---------------------------------------------------------
@register_model("focal")
class Focal(nn.Module):
    """FOCAL HRNet backbone + local/detect heads"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = FOCAL_HRNet()  # HRNet backbone
        self.local_head = UpsampleLocalHead(in_channels=32, out_channels=1, output_size=(512, 512))
        self.detect_head = DetectHead(in_channels=32, hidden_dim=256, out_dim=1)

        # Load pretrained weights
        weight_path = '/data/disk2/yer/ForensicHub/external/FOCAL/weights/FOCAL_HRNet_weights.pth'
        if os.path.isfile(weight_path):
            state_dict = torch.load(weight_path, map_location='cpu')
            try:
                self.backbone.load_state_dict(state_dict, strict=False)
                print(f"Loaded FOCAL_HRNet weights from {weight_path}")
            except Exception as e:
                print(f"Warning: Failed to fully load weights: {e}")
        else:
            print(f"Weight file not found: {weight_path}")

    def forward(self, image, mask, label, *args, **kwargs):
        label = label.float()  # [B] or [B,1]

        # Step 1: backbone forward
        feat = self.backbone(image)  # [B, 32, H/4, W/4]

        # Step 2: detect head
        pred_logits = self.detect_head(feat).squeeze(dim=1)  # [B]
        loss_label = F.binary_cross_entropy_with_logits(pred_logits, label)

        # Step 3: local head
        pred_masks = self.local_head(feat)  # [B, 1, 512, 512]
        loss_all = F.binary_cross_entropy_with_logits(pred_masks, mask.float(), reduction='none')
        loss_per_sample = loss_all.view(loss_all.size(0), -1).mean(dim=1)
        mask_valid = (mask.sum(dim=[1, 2, 3]) > 0).float()
        num_valid = mask_valid.sum()
        loss_mask = (loss_per_sample * mask_valid).sum() / (num_valid + 1e-6)

        return {
            "backward_loss": loss_label + loss_mask,
            "pred_mask": torch.sigmoid(pred_masks),
            "pred_label": torch.sigmoid(pred_logits),
            "visual_loss": {
                "loss_label": loss_label,
                "loss_mask": loss_mask,
            },
        }

# ---------------------------------------------------------
# Test run
# ---------------------------------------------------------
if __name__ == "__main__":
    import random
    random.seed(42)
    model = Focal().to(0)
    model.eval()

    B, C, H, W = 4, 3, 512, 512
    image = torch.randn(B, C, H, W).to(0)
    label = torch.ones(B).to(0)
    mask = torch.zeros(B, 1, H, W).to(0)
    for i in range(B):
        if random.random() > 0.5:
            mask[i, 0] = torch.randint(0, 2, (H, W)).float()

    with torch.no_grad():
        out = model(image, mask, label)

    print("== Forward Results ==")
    print(f"backward_loss: {out['backward_loss'].item():.4f}")
    print(f"loss_label:    {out['visual_loss']['loss_label'].item():.4f}")
    print(f"loss_mask:     {out['visual_loss']['loss_mask'].item():.4f}")
