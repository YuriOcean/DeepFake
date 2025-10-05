import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from ForensicHub.registry import register_model
import os


class UpsampleLocalHead(nn.Module):
    def __init__(self, in_channels, out_channels=1, up_factor=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.relu = nn.ReLU(inplace=True)
        self.out_conv = nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)
        self.up_factor = up_factor

    def forward(self, x, target_size=None):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.out_conv(x)
        # 上采样到 mask 尺寸或默认倍数
        if target_size is not None and target_size[0] > 0 and target_size[1] > 0:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        else:
            x = F.interpolate(x, scale_factor=self.up_factor, mode="bilinear", align_corners=False)
        return x

class DetectHead(nn.Module):
    def __init__(self, in_channels, hidden_dim=256, out_dim=1):
        super().__init__()
        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        self.pool_max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        avg = self.pool_avg(x).flatten(1)
        mx = self.pool_max(x).flatten(1)
        feat = torch.cat([avg, mx], dim=1)
        return self.mlp(feat)

# 只加载 HRNet backbone 的预训练权重
def load_backbone_only(backbone, weight_path):
    if not os.path.isfile(weight_path):
        print(f"[WARN] 预训练权重 {weight_path} 不存在，跳过加载。")
        return

    state = torch.load(weight_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]

    new_state = {}
    for k, v in state.items():
        nk = k.replace("module.", "")
        # 只保留 backbone 部分
        if nk.startswith("backbone.") or nk.startswith("conv1") or nk.startswith("layer"):
            new_state[nk] = v

    missing, unexpected = backbone.load_state_dict(new_state, strict=False)
    print(f"[INFO] Loaded HRNet backbone from {weight_path}")
    print(f"[INFO] Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

class FocalBackboneWrapper(nn.Module):
    def __init__(self, pretrained=True,
                 pretrained_path="/data/disk2/yer/ForensicHub/external/FOCAL/weights/HRNet_W48_C_ssld_pretrained.pth"):
        super().__init__()
        mod = importlib.import_module("external.FOCAL.models.hrnet")
        cls = getattr(mod, "FOCAL_HRNet")
        self.enc = cls()

        # 只加载 backbone 权重
        if pretrained and pretrained_path is not None:
            try:
                load_backbone_only(self.enc.net.backbone, pretrained_path)
            except Exception as e:
                print(f"[WARN] HRNet 预训练权重加载失败: {e}")

        self.out_channels = self._infer_out_channels()

    def _infer_out_channels(self):
        dummy = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            y = self.enc(dummy)
        if isinstance(y, (list, tuple)):
            y = y[-1]
        return y.shape[1]

    def forward(self, x):
        y = self.enc(x)
        if isinstance(y, (list, tuple)):
            y = y[-1]
        return y

@register_model("focal")
class BisaiFOCAL(nn.Module):
    def __init__(self, pretrained_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = FocalBackboneWrapper(pretrained=True, pretrained_path=pretrained_path)
        C = self.backbone.out_channels
        self.local_head = UpsampleLocalHead(in_channels=C, out_channels=1, up_factor=32)
        self.detect_head = DetectHead(in_channels=C, hidden_dim=256, out_dim=1)

    def forward(self, image, mask, label, *args, **kwargs):
        label = label.float()
        feat = self.backbone(image)


        pred_logits = self.detect_head(feat).squeeze(1)
        loss_label = F.binary_cross_entropy_with_logits(pred_logits, label)


        pred_masks = self.local_head(feat, target_size=mask.shape[-2:])
        loss_all = F.binary_cross_entropy_with_logits(pred_masks, mask.float(), reduction="none")


        loss_per_sample = loss_all.view(loss_all.size(0), -1).mean(dim=1)


        mask_valid = torch.ones(mask.size(0), device=mask.device)
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


if __name__ == "__main__":
    import random
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BisaiFOCAL(pretrained_path=None).to(device).eval()

    B, C, H, W = 2, 3, 512, 512
    x = torch.randn(B, C, H, W, device=device)
    y_label = torch.ones(B, device=device)
    y_mask = torch.zeros(B, 1, H, W, device=device)
    for i in range(B):
        if random.random() > 0.5:
            y_mask[i, 0] = torch.randint(0, 2, (H, W), device=device).float()

    with torch.no_grad():
        out = model(x, y_mask, y_label)

    print("OK shapes:", out["pred_mask"].shape, out["pred_label"].shape)
    print("Losses:", {k: float(v) for k, v in out["visual_loss"].items()})
