import os
from typing import Dict, List
import numpy as np
from PIL import Image
import torch
import pandas as pd

from ForensicHub.core.base_dataset import BaseDataset
from ForensicHub.registry import register_dataset


def _as_rgb(im: Image.Image) -> Image.Image:
    if im.mode == "RGB":
        return im
    return im.convert("RGB")


def _read_mask(path: str, size: int) -> np.ndarray:
    if not os.path.isfile(path):
        return np.zeros((size, size), dtype=np.uint8)
    m = Image.open(path).convert("L")
    m = (np.array(m) > 0).astype(np.uint8)
    m = Image.fromarray(m)
    m = m.resize((size, size), Image.NEAREST)
    return np.array(m).astype(np.uint8)


@register_dataset("FFPPDataset")
class FFPPDataset(BaseDataset):

    def __init__(self, path: str, root_dir: str, image_size: int = 512, mask_dir: str = None, **kwargs):
        self.root_dir = root_dir
        self.csv_path = path
        self.image_size = image_size
        self.mask_dir = mask_dir

        super().__init__(path=root_dir, **kwargs)

    def _init_dataset_path(self):
        if not os.path.isfile(self.csv_path):
            raise RuntimeError(f"Annotations CSV not found: {self.csv_path}")
        df = pd.read_csv(self.csv_path)

        samples = []
        for _, row in df.iterrows():
            frame_path = os.path.join(self.root_dir, row["frame_path"])
            if not os.path.isfile(frame_path):
                continue
            label_val = int(row["label"])
            mask_path = None
            if self.mask_dir:
                mask_path_candidate = os.path.join(self.mask_dir, os.path.basename(frame_path))
                if os.path.isfile(mask_path_candidate):
                    mask_path = mask_path_candidate
            samples.append({
                "img_path": frame_path,
                "mask_path": mask_path,
                "has_mask": mask_path is not None,
                "label": label_val,
                "id": os.path.splitext(os.path.basename(frame_path))[0],
            })

        if len(samples) == 0:
            raise RuntimeError("No samples found in FFPP dataset")

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]

        # 读取图像
        img = _as_rgb(Image.open(s["img_path"]))
        img = img.resize((self.image_size, self.image_size))
        image = np.array(img)

        # 读取 mask
        if s["has_mask"] and s["mask_path"]:
            mask_np = _read_mask(s["mask_path"], self.image_size)
        else:
            mask_np = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        # 图像级 label
        label = torch.tensor(s["label"], dtype=torch.float)

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
