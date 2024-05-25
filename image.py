import dataclasses

from PIL import ImageEnhance


def adjust_contrast(contrast, image):
    enhancer = ImageEnhance.Contrast(image)
    im_contrast = enhancer.enhance(contrast)
    return im_contrast


@dataclasses.dataclass
class Image:
    filename: str
    path: str