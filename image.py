import math

import numpy as np
import tensorflow as tf

from PIL import ImageEnhance


def adjust_contrast(contrast, image):
    enhancer = ImageEnhance.Contrast(image)
    image_contrast = enhancer.enhance(contrast)
    return image_contrast


def next_patch_horizontal(spec, n):
    step = math.floor(spec.shape[0] / n)
    for i in range(n):
        yield spec[i * step:(i + 1) * step, :, :]


def next_patch_vertical(spec, n):
    step = math.floor(spec.shape[1] / n)
    for i in range(n):
        yield spec[:, i * step:(i + 1) * step, :]


def get_color_mode(colormode):
    return 3 if colormode == 'RGB' else 1


def get_input_shape(colormode, patches: int, orientation: int, spec_height: int, spec_width: int) -> tuple[int, int, int]:
    match orientation:
        case 'horizontal':
            return math.floor(spec_height / patches), spec_width, get_color_mode(colormode)
        case 'vertical':
            return spec_height, math.floor(spec_width / patches), get_color_mode(colormode)
        case 'horizontal+vertical':
            return math.floor(spec_height / patches), math.floor(spec_width / patches), get_color_mode(colormode)
        case _:
            raise ValueError('orientation invalid')
