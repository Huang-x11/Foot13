"""Uniform image-only augmentation for Foot13, including all background."""

from mmpose.datasets.transforms import Albumentation
from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class Foot13PhotometricAug(Albumentation):
    """Apply mild brightness/contrast and one box blur without moving labels.

    """

    def __init__(self, brightness_limit=0.1, contrast_limit=0.1,
                 brightness_contrast_prob=0.3, blur_kernel=3, blur_prob=0.1):
        super().__init__(transforms=[
            dict(type='RandomBrightnessContrast',
                 brightness_limit=brightness_limit, contrast_limit=contrast_limit,
                 brightness_by_max=True, p=brightness_contrast_prob),
            dict(type='Blur', blur_limit=(blur_kernel, blur_kernel), p=blur_prob),
        ])

