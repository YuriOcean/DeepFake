import os
from typing import Dict, List
import numpy as np
from PIL import Image
import torch

from ForensicHub.core.base_dataset import BaseDataset
from ForensicHub.registry import register_dataset


def _as_rgb(im: Image.Image) -> Image.Image:
    """确保图像为 RGB"""
    if im.mode == "RGB":
        return im
    return im.convert("RGB")


def _read_mask(path: str, size: int) -> np.ndarray:
    """读取 mask，先二值化再 resize"""
    m = Image.open(path).convert("L")
    m = (np.array(m) > 0).astype(np.uint8)  # 二值化
    m = Image.fromarray(m)
    m = m.resize((size, size), Image.NEAREST)
    return np.array(m).astype(np.uint8)


def _list_files(dirpath: str, exts: List[str] = [".png", ".jpg", ".tif"]) -> List[str]:
    """列出指定后缀的文件"""
    if not os.path.isdir(dirpath):
        return []
    return sorted([f for f in os.listdir(dirpath) if any(f.lower().endswith(e) for e in exts)])


@register_dataset("CoverageDataset")
class CoverageDataset(BaseDataset):
    """覆盖篡改区域的二分类与 mask 数据集"""

    MASK_EXTS = [".png", ".jpg", ".tif"]  # 自动匹配 mask 后缀

    def __init__(self, root_dir: str, image_size: int = 512, mask_size: int = None, mask_dir_name: str = "mask", **kwargs):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "tamper")
        self.mask_dir = os.path.join(root_dir, mask_dir_name)
        self.image_size = image_size
        self.mask_size = mask_size or image_size  # 默认和输入图像大小一致

        super().__init__(path=root_dir, **kwargs)

    def _init_dataset_path(self):
        image_files = _list_files(self.image_dir)
        if len(image_files) == 0:
            raise RuntimeError(f"No images found in {self.image_dir}")

        samples = []
        for fname in image_files:
            img_path = os.path.join(self.image_dir, fname)

            # 自动匹配 mask 文件
            mask_path = None
            has_mask = False
            image_name = os.path.splitext(fname)[0]
            for ext in self.MASK_EXTS:
                candidate = os.path.join(self.mask_dir, image_name + ext)
                if os.path.isfile(candidate):
                    mask_path = candidate
                    has_mask = True
                    break

            samples.append({
                "img_path": img_path,
                "mask_path": mask_path,
                "has_mask": has_mask,
                "id": image_name,
            })

        if len(samples) == 0:
            raise RuntimeError("No samples found")

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]

        # 读取图像
        img = _as_rgb(Image.open(s["img_path"]))
        img = img.resize((self.image_size, self.image_size))
        image = np.array(img)
        print(f"[DEBUG] mask_path={s['mask_path']}")  # 打印 mask 路径

        # 读取 mask
        if s["has_mask"] and s["mask_path"]:
            mask_img = Image.open(s["mask_path"]).convert("L")
            mask_np = (np.array(mask_img) > 0).astype(np.uint8)  # 二值化
            mask_img = Image.fromarray(mask_np)
            mask_img = mask_img.resize((self.mask_size, self.mask_size), Image.NEAREST)
            mask_np = np.array(mask_img).astype(np.uint8)
        else:
            mask_np = np.zeros((self.mask_size, self.mask_size), dtype=np.uint8)

        # 二分类 label
        label_val = 1 if mask_np.sum() > 0 else 0
        label = torch.tensor(label_val, dtype=torch.float)

        # 数据增强
        if getattr(self, "common_transform", None):
            image = self.common_transform(image=image)["image"]
        if getattr(self, "post_transform", None):
            out = self.post_transform(image=image, mask=mask_np)
            image, mask_np = out["image"], out["mask"]

        # 转成 Tensor
        if isinstance(mask_np, torch.Tensor):
            mask = mask_np.long().unsqueeze(0)
        else:
            mask = torch.from_numpy(mask_np).long().unsqueeze(0)

        return {
            "image": image,
            "label": label,
            "mask": mask,
        }