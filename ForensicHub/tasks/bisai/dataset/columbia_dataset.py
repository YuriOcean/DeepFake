import os
from typing import Dict, List, Optional, Tuple
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
    m = Image.open(path)
    m = (np.array(m) > 0).astype(np.uint8)     # 二值化
    m = Image.fromarray(m)
    m = m.resize((size, size), Image.NEAREST)
    return np.array(m).astype(np.uint8)


def _list_tifs(dirpath: str) -> List[str]:
    if not os.path.isdir(dirpath):
        return []
    return sorted(
        [f for f in os.listdir(dirpath) if f.lower().endswith(".tif") and f.lower() != "thumbs.db"]
    )

@register_dataset("ColumbiaDataset")
class ColumbiaDataset(BaseDataset):


    def __init__(self,
                 root_dir: str,
                 image_size: int = 512,
                 include_original: bool = True,
                 split_list: Optional[str] = None,
                 **kwargs):
        self.root_dir = root_dir

        self.auth_dir = self._pick_exist(
            os.path.join(root_dir, "4cam_auth", "4camauth"),
            os.path.join(root_dir, "4cam_auth"),
        )
        self.splc_dir = self._pick_exist(
            os.path.join(root_dir, "4cam_splc", "4camsplc"),
            os.path.join(root_dir, "4cam_splc"),
        )

        self.image_size = image_size
        self.include_original = include_original
        self.split_list = split_list

        super().__init__(path=root_dir, **kwargs)

    # ------- BaseDataset------- #
    def _init_dataset_path(self) -> None:
        allow_set: Optional[set] = None
        if self.split_list and os.path.isfile(self.split_list):
            with open(self.split_list, "r") as f:
                allow_set = {ln.strip() for ln in f if ln.strip()}

        samples: List[Dict] = []

        if self.include_original and self.auth_dir and os.path.isdir(self.auth_dir):
            for fname in _list_tifs(self.auth_dir):
                if allow_set is not None and fname not in allow_set:
                    continue
                samples.append({
                    "img_path": os.path.join(self.auth_dir, fname),
                    "label": 0,
                    "mask_path": None,
                    "has_mask": False,
                    "id": os.path.splitext(fname)[0],
                })

        if self.splc_dir and os.path.isdir(self.splc_dir):
            for fname in _list_tifs(self.splc_dir):
                if allow_set is not None and fname not in allow_set:
                    continue
                img_path = os.path.join(self.splc_dir, fname)
                mask_path = self._find_mask(img_path, fname)
                has_mask = mask_path is not None
                samples.append({
                    "img_path": img_path,
                    "label": 1,
                    "mask_path": mask_path,
                    "has_mask": has_mask,
                    "id": os.path.splitext(fname)[0],
                })

        if len(samples) == 0:
            raise RuntimeError(
                f"No samples found under:\n  auth_dir={self.auth_dir}\n  splc_dir={self.splc_dir}\n"
                f"Check folder structure or split_list."
            )

        self.samples = samples
        self.entry_path = self.path

    # ------- Dataset 接口 ------- #
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]


        img = _as_rgb(Image.open(s["img_path"]))
        img = img.resize((self.image_size, self.image_size))
        image = np.array(img)

        # 读 mask
        if s["has_mask"] and s["mask_path"] and os.path.isfile(s["mask_path"]):
            mask_np = _read_mask(s["mask_path"], self.image_size)
        else:
            mask_np = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        # 预处理/增强管线
        if self.common_transform:
            image = self.common_transform(image=image)['image']
        if self.post_transform:
            out = self.post_transform(image=image, mask=mask_np)
            image, mask_np = out['image'], out['mask']


        if isinstance(mask_np, np.ndarray):
            mask = torch.from_numpy(mask_np.copy()).long().unsqueeze(0)
        else:
            mask = mask_np.long().unsqueeze(0)

        return {
            "image": image,
            "label": torch.tensor(s["label"], dtype=torch.float),
            "mask": mask,
        }

    @staticmethod
    def _pick_exist(*candidates: str) -> Optional[str]:
        for p in candidates:
            if os.path.isdir(p):
                return p
        return None

    def _find_mask(self, img_path: str, fname: str) -> Optional[str]:

        img_dir = os.path.dirname(img_path)
        root = self.root_dir

        candidate_dirs = [
            os.path.join(img_dir, "edgemask"),
            os.path.join(root, "4cam_splc", "edgemask"),
            os.path.join(root, "edgemask"),
        ]

        if self.auth_dir:
            candidate_dirs.append(os.path.join(self.auth_dir, "edgemask"))

        base, ext = os.path.splitext(fname)
        exts = [ext.lower(), ".tif", ".png", ".bmp"]

        for d in candidate_dirs:
            if not os.path.isdir(d):
                continue
            for e in exts:
                cand = os.path.join(d, base + e)
                if os.path.isfile(cand):
                    return cand
        return None
