import numpy as np
import tensorflow as tf

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


def extract_features(features, image_sliced, model, n_patches, orientation, preprocess_input, spec):
    match orientation:
        case 'horizontal':
            extract_features_horizontal(features, image_sliced, model, n_patches, preprocess_input, spec)
        case 'vertical':
            extract_features_vertical(features, image_sliced, model, n_patches, preprocess_input, spec)
        case 'horizontal+vertical':
            extract_features_horizontal_vertical(features, image_sliced, model, n_patches, preprocess_input, spec)
        case _:
            raise ValueError('orientation must be horizontal, vertical or horizontal+vertical')


def extract(features, image_sliced, model, p, preprocess_input):
    p = preprocess_input(p)
    image_sliced.append(tf.keras.preprocessing.image.array_to_img(p))
    p = np.expand_dims(p, axis=0)
    features.append(model.predict(p))


def extract_features_horizontal_vertical(features, image_sliced, model, n_patches, preprocess_input, spec):
    for p in next_patch_horizontal(spec, n_patches):
        for q in next_patch_vertical(p, n_patches):
            extract(features, image_sliced, model, q, preprocess_input)


def extract_features_vertical(features, image_sliced, model, n_patches, preprocess_input, spec):
    for p in next_patch_vertical(spec, n_patches):
        extract(features, image_sliced, model, p, preprocess_input)


def extract_features_horizontal(features, image_sliced, model, n_patches, preprocess_input, spec):
    for p in next_patch_horizontal(spec, n_patches):
        extract(features, image_sliced, model, p, preprocess_input)
