import glob
import math
import numpy as np
import os
import pathlib

import pandas as pd
import tensorflow as tf


from PIL import Image, ImageEnhance

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PATHBASE = '/home/xandao/Imagens'
PATCHES = [3]


def next_patch(spec, n):
    step = math.floor(spec.shape[0] / n)
    for i in range(n):
        yield spec[i * step:(i + 1) * step, :, :]


def get_model(model, **kwargs):
    if model == 'vgg16':
        return tf.keras.applications.vgg16.VGG16(**kwargs), tf.keras.applications.vgg16.preprocess_input
    if model == 'resnet50v2':
        return tf.keras.applications.resnet_v2.ResNet50V2(**kwargs), tf.keras.applications.resnet_v2.preprocess_input
    if model == 'mobilenetv2':
        return tf.keras.applications.mobilenet_v2.MobileNetV2(
            **kwargs), tf.keras.applications.mobilenet_v2.preprocess_input

    raise ValueError


def adjust_contrast(image):
    #image brightness enhancer
    enhancer = ImageEnhance.Contrast(image)
            
    factor = 1.5
    image = enhancer.enhance(factor)
    return image


def extract_features(cnn, color, dataset, gpuid, folds, image_size, input_path, level, minimum_image, output_path, patches, region):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)
    spec_height = image_size[0]
    spec_width = image_size[1]
    input_path_proto = os.path.join(input_path, 'f%d', '*.jpeg')

    for n_patches in patches:
        print('Slicing images into %d non-overlapping patches...' % (n_patches))
        tf.keras.backend.clear_session()

        input_shape = (math.floor(spec_height / n_patches), spec_width, 3)

        model, preprocess_input = get_model(cnn, weights='imagenet', include_top=False,
                                            input_shape=input_shape, pooling='avg')
        total_samples=0
        imgs_sliced = []
        for fold in folds:
            print('Extracting features for fold %d...' % (fold))
            if len(glob.glob(input_path_proto % (fold))) == 0:
                raise RuntimeError('No files found in: %s' % (input_path_proto % (fold)))

            features = []
            for fname in sorted(glob.glob(input_path_proto % (fold))):
                print('fname: %s' % fname)
                img = tf.keras.preprocessing.image.load_img(fname)
                img = adjust_contrast(img)     
                spec = tf.keras.preprocessing.image.img_to_array(img)
                for p in next_patch(spec, n_patches):
                    p = preprocess_input(p)
                    imgs_sliced.append(tf.keras.preprocessing.image.array_to_img(p))
                    p = np.expand_dims(p, axis=0)
                    features.append(model.predict(p))

            features = np.concatenate(features)
            save_file('npy', features, fold, n_patches, output_path)
            save_file('npz', features, fold, n_patches, output_path)
            n_samples, n_features = features.shape
            for i, img in enumerate(imgs_sliced):
                tf.keras.preprocessing.image.save_img(f'{i}.png', img)
            total_samples+=n_samples
        # save_information(color, cnn, dataset, image_size, input_path, level, minimum_image, n_features, output_path, n_patches, region, total_samples)


def save_file(extension, features, fold, n_patches, output_path):
    output_path = os.path.join(output_path, extension)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_path = os.path.join(output_path, 'fold-%d_patches-%d.' + extension)
    output_filename = output_path % (fold, n_patches)
    if extension == 'npy':
        np.save(output_filename, features, allow_pickle=True)
    else:
        np.savez_compressed(output_filename, x=features, y=np.repeat(fold, features.shape[0]))


def save_information(color, cnn, dataset, image_size, input_path, level, minimum_image, n_features, output_path, patch, region, total_samples):
    height = image_size[0]
    width = image_size[1]
    index = ['cnn', 'color', 'dataset', 'height', 'width', 'level', 'minimum_image', 'input_path', 'output_path',
             'patch', 'n_features', 'total_samples']
    data = [cnn, color, dataset, height, width, level, minimum_image, input_path, output_path, patch, n_features, total_samples]

    if region:
        index.append('region')
        data.append(region)

    df = pd.DataFrame(data, index=index)
    filename = os.path.join(output_path, 'info.csv')
    df.to_csv(filename, sep=';', index=index, header=None, line_terminator='\n', doublequote=True)


def prepare(cnn, color, dataset, image_size, level, minimum_image, input_path, region=None):
    if not os.path.exists(input_path):
        raise SystemError('path (%s) not exists' % input_path)

    list_folders = [f for f in pathlib.Path(input_path).glob('*') if f.is_dir()]
    folds = len(list_folders)
    folds = list(range(1, folds + 1))
    image_size = (int(image_size), int(image_size))
    gpuid = 0
    patches = PATCHES
    patches = list(patches)

    features_folder = dataset + '_features'
    if region:
        output_path = os.path.join(PATHBASE, features_folder, color, str(image_size[0]), level, region, minimum_image, cnn)
    else:
        output_path = os.path.join(PATHBASE, features_folder, color, str(image_size[0]), level, minimum_image, cnn)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print('Feature Extraction Parameters')
    print('Pre-trained model: %s' % cnn)
    print('Non-overlapping patches per image: %s' % str(patches))
    print('Folds: %s' % str(folds))
    print('Image Dimensions h=%s, w=%s ' % (image_size, image_size))
    print('Format string for input: %s ' % input_path)
    print('Format string for output: %s ' % output_path)
    print('GPU ID: %d' % gpuid)

    extract_features(cnn, color, dataset, gpuid, folds, image_size, input_path, level, minimum_image, output_path, patches, region)


def main():
    list_color = ['RGB', 'GRAYSCALE']
    list_size = ['256', '400', '512']
    list_minimum_image = ['20', '10', '5']
    list_cnn = ['vgg16', 'resnet50v2', 'mobilenetv2']
    list_dataset = ['pr_dataset']
    list_level = ['specific_epithet_trusted']
    list_region = ['Norte', 'Nordeste', 'Sul', 'Sudeste', 'Centro-Oeste']
    for cnn in list_cnn:
        for color in list_color:
            for image_size in list_size:
                for minimum_image in list_minimum_image:
                    for level in list_level:
                        for dataset in list_dataset:
                            print('cnn: %s color: %s dataset: %s image_size: %s level: %s minimum_image: %s '
                                  % (cnn, color, dataset, image_size, level, minimum_image))
                            if 'regions_dataset' == dataset:
                                for region in list_region:
                                    path = os.path.join(PATHBASE, dataset, color, level, region, image_size,
                                                        minimum_image)
                                    prepare(cnn, color, dataset, image_size, level, minimum_image, path, region=region)
                            else:
                                path = os.path.join(PATHBASE, dataset, color, level, image_size, minimum_image)
                                prepare(cnn, color, dataset, image_size, level, minimum_image, path)
                                



if __name__ == '__main__':
    main()
