import albumentations as albu
from albumentations.pytorch import ToTensorV2
from ForensicHub.core.base_transform import BaseTransform
from ForensicHub.registry import register_transform


@register_transform("FocalTransform")
class FocalTransform(BaseTransform):
    """Minimal Transform for pixel-level tasks (no data augmentation)."""

    def __init__(self, output_size: tuple = (512, 512), norm_type='image_net'):
        super().__init__()
        self.output_size = output_size
        self.norm_type = norm_type

    def get_post_transform(self) -> albu.Compose:
        """Normalization + tensor conversion for image and mask."""
        if self.norm_type == 'image_net':
            return albu.Compose([
                albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(transpose_mask=True)
            ])
        elif self.norm_type == 'clip':
            return albu.Compose([
                albu.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
                ToTensorV2(transpose_mask=True)
            ])
        elif self.norm_type == 'standard':
            return albu.Compose([
                albu.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(transpose_mask=True)
            ])
        elif self.norm_type == 'none':
            return albu.Compose([
                albu.ToFloat(max_value=255.0),
                ToTensorV2(transpose_mask=True)
            ])
        else:
            raise NotImplementedError("Normalization type not supported: use image_net, clip, standard, or none")

    def get_train_transform(self) -> albu.Compose:
        """Minimal train transform: only resize + post transform."""
        return albu.Compose([
            albu.Resize(height=self.output_size[0], width=self.output_size[1], interpolation=1),  # image: bilinear
            albu.Resize(height=self.output_size[0], width=self.output_size[1], interpolation=0),  # mask: nearest
        ], additional_targets={'mask': 'mask'})

    def get_test_transform(self) -> albu.Compose:
        """Minimal test transform: same as train."""
        return self.get_train_transform()
