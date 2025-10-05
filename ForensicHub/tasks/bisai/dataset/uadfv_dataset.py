import os
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch

from ForensicHub.core.base_dataset import BaseDataset
from ForensicHub.registry import register_dataset

def _as_rgb(im: Image.Image) -> Image.Image:
    return im if im.mode == "RGB" else im.convert("RGB")

def _read_mask_zeros(size: int) -> np.ndarray:
    return np.zeros((size, size), dtype=np.uint8)

def _list_dir(path: str) -> List[str]:
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]) if os.path.isdir(path) else []

def _list_imgs(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [f for f in os.listdir(path) if os.path.splitext(f)[1].lower() in exts]
    return sorted(files)

def _safe_ints(s: str) -> List[float]:
    return [float(x) for x in re.findall(r"-?\d+\.?\d*", s)]

def _read_landmarks(txt_or_npy: str) -> Optional[np.ndarray]:
    if not os.path.isfile(txt_or_npy):
        return None
    ext = os.path.splitext(txt_or_npy)[1].lower()
    try:
        if ext == ".npy":
            arr = np.load(txt_or_npy)
            if arr.ndim == 2 and arr.shape[-1] == 2:
                return arr.astype(np.float32)
            if arr.ndim == 2 and arr.shape[0] == 2:
                return arr.T.astype(np.float32)
            return None
        else:
            pts: List[List[float]] = []
            with open(txt_or_npy, "r") as f:
                for ln in f:
                    nums = _safe_ints(ln)
                    if len(nums) >= 2:
                        pts.append([nums[0], nums[1]])
            return np.array(pts, dtype=np.float32) if len(pts) >= 3 else None
    except Exception:
        return None

def _bbox_from_landmarks(landmarks: np.ndarray, expand: float = 1.3) -> Optional[Tuple[int,int,int,int]]:
    if landmarks is None or landmarks.size == 0:
        return None
    xs, ys = landmarks[:,0], landmarks[:,1]
    cx, cy = (xs.min()+xs.max())/2.0, (ys.min()+ys.max())/2.0
    w, h = max(xs.max()-xs.min(),1.0), max(ys.max()-ys.min(),1.0)
    side = max(w,h)*expand
    x0 = int(round(cx-side/2.0))
    y0 = int(round(cy-side/2.0))
    x1 = int(round(cx+side/2.0))
    y1 = int(round(cy+side/2.0))
    return x0, y0, x1, y1

def _crop_with_pad(img: Image.Image, box: Tuple[int,int,int,int]) -> Image.Image:
    return img.crop(box)

def _pick_exist(*candidates: str) -> Optional[str]:
    for p in candidates:
        if os.path.isdir(p):
            return p
    return None

# ---------------------- Dataset ---------------------- #
@register_dataset("UADFVDataset")
class UADFVDataset(BaseDataset):

    def __init__(self,
                 root_dir: str,
                 image_size: int = 512,
                 include_fake: bool = True,
                 include_real: bool = True,
                 frames_per_video: Optional[int] = None,
                 sampling: str = "uniform",
                 crop_by_landmarks: bool = False,
                 bbox_expand: float = 1.3,
                 split_list: Optional[str] = None,
                 **kwargs):
        self.root_dir = root_dir
        self.image_size = image_size
        self.include_fake = include_fake
        self.include_real = include_real
        self.frames_per_video = frames_per_video
        self.sampling = sampling
        self.crop_by_landmarks = crop_by_landmarks
        self.bbox_expand = bbox_expand
        self.split_list = split_list

        self.fake_frames_root = _pick_exist(os.path.join(root_dir,"fake","frames"))
        self.fake_lm_root = _pick_exist(os.path.join(root_dir,"fake","landmarks"))
        self.real_frames_root = _pick_exist(os.path.join(root_dir,"real","frames"))
        self.real_lm_root = _pick_exist(os.path.join(root_dir,"real","landmarks"))

        super().__init__(path=root_dir, **kwargs)

    def _init_dataset_path(self) -> None:
        allow_vids: Optional[set] = None
        if self.split_list and os.path.isfile(self.split_list):
            with open(self.split_list,"r") as f:
                allow_vids = {ln.strip() for ln in f if ln.strip()}

        samples: List[Dict] = []

        def collect(frames_root: str, lm_root: str, label: int):
            if not frames_root:
                return
            for vid in _list_dir(frames_root):
                if allow_vids is not None and vid not in allow_vids:
                    continue
                frame_dir = os.path.join(frames_root, vid)
                imgs = _list_imgs(frame_dir)
                if len(imgs) == 0:
                    continue
                picked = self._pick_indices(len(imgs))
                for idx in picked:
                    fname = imgs[idx]
                    lm_path = self._match_landmark(lm_root, vid, fname)
                    samples.append({
                        "img_path": os.path.join(frame_dir, fname),
                        "lm_path": lm_path,
                        "label": label,
                        "vid": vid,
                        "frame_name": os.path.splitext(fname)[0],
                    })

        if self.include_fake:
            collect(self.fake_frames_root, self.fake_lm_root, 1)
        if self.include_real:
            collect(self.real_frames_root, self.real_lm_root, 0)

        if len(samples) == 0:
            raise RuntimeError(
                f"UADFVDataset: no samples found under:\n"
                f"  fake frames: {self.fake_frames_root}\n"
                f"  real frames: {self.real_frames_root}\n"
            )
        self.samples = samples
        self.entry_path = self.path

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        img = _as_rgb(Image.open(s["img_path"]))

        if self.crop_by_landmarks and s["lm_path"]:
            lms = _read_landmarks(s["lm_path"])
            box = _bbox_from_landmarks(lms, expand=self.bbox_expand) if lms is not None else None
            if box:
                img = _crop_with_pad(img, box)

        img = img.resize((self.image_size, self.image_size))
        image = np.array(img)

        mask_np = _read_mask_zeros(self.image_size)

        if self.common_transform:
            image = self.common_transform(image=image)['image']
        if self.post_transform:
            out = self.post_transform(image=image, mask=mask_np)
            image, mask_np = out['image'], out['mask']

        mask = mask_np.clone().long().unsqueeze(0)

        return {
            "image": image,
            "label": torch.tensor(s["label"], dtype=torch.float),
            "mask": mask,
        }

    def _pick_indices(self, n: int) -> List[int]:
        if self.frames_per_video is None or self.frames_per_video <= 0 or self.frames_per_video >= n:
            return list(range(n))
        k = self.frames_per_video
        if self.sampling == "uniform":
            idxs = np.linspace(0, n - 1, num=k, dtype=int).tolist()
            seen, dedup = set(), []
            for i in idxs:
                if i not in seen:
                    seen.add(i)
                    dedup.append(i)
            cur = 0
            while len(dedup) < k and cur < n:
                if cur not in seen:
                    seen.add(cur)
                    dedup.append(cur)
                cur += 1
            return dedup[:k]
        if self.sampling == "head":
            return list(range(min(k,n)))
        if self.sampling == "tail":
            return list(range(max(0,n-k), n))
        if self.sampling == "middle":
            start = max(0, (n-k)//2)
            return list(range(start, min(start+k, n)))
        return np.linspace(0, n-1, num=k, dtype=int).tolist()

    def _match_landmark(self, lm_root: str, vid: str, frame_fname: str) -> Optional[str]:
        if not lm_root:
            return None
        name, _ = os.path.splitext(frame_fname)
        cand_dir = os.path.join(lm_root, vid)
        if not os.path.isdir(cand_dir):
            return None
        for ext in (".txt", ".pts", ".npy"):
            cand = os.path.join(cand_dir, name + ext)
            if os.path.isfile(cand):
                return cand
        return None
