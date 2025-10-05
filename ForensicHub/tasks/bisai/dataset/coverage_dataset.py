import os
from typing import Dict, List
import numpy as np
from PIL import Image
import torch

from ForensicHub.core.base_dataset import BaseDataset
from ForensicHub.registry import register_dataset


def _as_rgb(im: Image.Image) -> Image.Image:
    if im.mode == "RGB":
        return im
    return im.convert("RGB")


def _read_mask(path: str, size: int) -> np.ndarray:
    m = Image.open(path).convert("L")
    m = (np.array(m) > 0).astype(np.uint8)
    m = Image.fromarray(m)
    m = m.resize((size, size), Image.NEAREST)
    return np.array(m).astype(np.uint8)


def _list_files(dirpath: str, exts: List[str] = [".png", ".jpg", ".tif"]) -> List[str]:
    if not os.path.isdir(dirpath):
        return []
    return sorted([f for f in os.listdir(dirpath) if any(f.lower().endswith(e) for e in exts)])


@register_dataset("CoverageDataset")
class CoverageDataset(BaseDataset):


    def __init__(self, root_dir: str, image_size: int = 512, mask_dir_name: str = "mask", **kwargs):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "tamper")
        self.mask_dir = os.path.join(root_dir, mask_dir_name)
        self.image_size = image_size

        super().__init__(path=root_dir, **kwargs)

    def _init_dataset_path(self):
        image_files = _list_files(self.image_dir)
        mask_files = _list_files(self.mask_dir)

        if len(image_files) == 0:
            raise RuntimeError(f"No images found in {self.image_dir}")

        samples = []
        for fname in image_files:
            img_path = os.path.join(self.image_dir, fname)
            mask_path = os.path.join(self.mask_dir, fname)
            has_mask = os.path.isfile(mask_path)
            samples.append({
                "img_path": img_path,
                "mask_path": mask_path if has_mask else None,
                "has_mask": has_mask,
                "id": os.path.splitext(fname)[0],
            })

        if len(samples) == 0:
            raise RuntimeError("No samples found")

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]


        img = _as_rgb(Image.open(s["img_path"]))
        img = img.resize((self.image_size, self.image_size))
        image = np.array(img)

        # 读取 mask
        if s["has_mask"] and s["mask_path"]:
            mask_np = _read_mask(s["mask_path"], self.image_size)
        else:
            mask_np = np.zeros((self.image_size, self.image_size), dtype=np.uint8)


        label_val = 1 if mask_np.sum() > 0 else 0
        label = torch.tensor(label_val, dtype=torch.float)

        # 数据增强
        if self.common_transform:
            image = self.common_transform(image=image)["image"]
        if self.post_transform:
            out = self.post_transform(image=image, mask=mask_np)
            image, mask_np = out["image"], out["mask"]

        mask = torch.tensor(mask_np, dtype=torch.long).unsqueeze(0)

        return {
            "image": image,
            "label": label,
            "mask": mask,
        }
