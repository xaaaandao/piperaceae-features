import numpy as np
import tensorflow as tf

from typing import Any

from image import next_patch_horizontal, next_patch_vertical


def get_model(model: str, **kwargs):
    match model:
        case 'vgg16':
            return tf.keras.applications.vgg16.VGG16(**kwargs), tf.keras.applications.vgg16.preprocess_input
        case 'resnet50v2':
            return tf.keras.applications.resnet_v2.ResNet50V2(
                **kwargs), tf.keras.applications.resnet_v2.preprocess_input
        case 'mobilenetv2':
            return tf.keras.applications.mobilenet_v2.MobileNetV2(
                **kwargs), tf.keras.applications.mobilenet_v2.preprocess_input
        case _:
            raise ValueError('Unknown model')


def extract_features(features: list, images: list, model: Any, n_patches: int, orientation: str,
                     preprocess_input: Any, spec: Any) -> None:
    match orientation:
        case 'horizontal':
            extract_features_horizontal(features, images, model, n_patches, preprocess_input, spec)
        case 'vertical':
            extract_features_vertical(features, images, model, n_patches, preprocess_input, spec)
        case 'horizontal+vertical':
            extract_features_horizontal_vertical(features, images, model, n_patches, preprocess_input, spec)
        case _:
            raise ValueError('orientation must be horizontal, vertical or horizontal+vertical')


def extract(features: list, images: list, model: Any, patch: np.ndarray, preprocess_input: Any) -> None:
    patch = preprocess_input(patch)
    images.append(tf.keras.preprocessing.image.array_to_img(patch))
    patch = np.expand_dims(patch, axis=0)
    features.append(model.predict(patch))


def extract_features_horizontal_vertical(features: list, images: list, model: Any, n_patches: int,
                                         preprocess_input: Any, spec: Any):
    for p in next_patch_horizontal(spec, n_patches):
        for q in next_patch_vertical(p, n_patches):
            extract(features, images, model, q, preprocess_input)


def extract_features_vertical(features: list, images: list, model: Any, n_patches: int, preprocess_input: Any,
                              spec: Any):
    for p in next_patch_vertical(spec, n_patches):
        extract(features, images, model, p, preprocess_input)


def extract_features_horizontal(features, image_sliced, model, n_patches, preprocess_input, spec):
    for p in next_patch_horizontal(spec, n_patches):
        extract(features, image_sliced, model, p, preprocess_input)
