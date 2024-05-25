import math
from typing import Tuple

import tensorflow as tf


def get_model(model, **kwargs):
    match model:
        case 'vgg16':
            return tf.keras.applications.vgg16.VGG16(**kwargs), tf.keras.applications.vgg16.preprocess_input
        case 'resnet50v2':
            return tf.keras.applications.resnet_v2.ResNet50V2(
                **kwargs), tf.keras.applications.resnet_v2.preprocess_input
        case 'mobilenetv2':
            return tf.keras.applications.mobilenet_v2.MobileNetV2(
                **kwargs), tf.keras.applications.mobilenet_v2.preprocess_input

    raise ValueError


def get_input_shape(orientation: str, patch: int, spec_height: int, spec_width: int)-> Tuple:
    match orientation:
        case 'horizontal':
            return math.floor(spec_height / patch), spec_width, 3
        case 'vertical':
            return spec_height, math.floor(spec_width / patch), 3
