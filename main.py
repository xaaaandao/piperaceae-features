import click
import os
import pathlib

import numpy as np
import tensorflow as tf

from cnn import get_model, extract_features
from image import adjust_contrast, get_input_shape
from save import save_image, save_file, save_information

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


@click.command()
@click.option('--colormode', type=click.Choice(['RGB', 'grayscale']), required=True)
@click.option('--contrast', type=float, required=False, default=-1)
@click.option('--folds', '-f', type=int, default=3)
@click.option('--gpu', type=int, default=0)
@click.option('--height', '-h', type=int, required=True)
@click.option('--input', '-i', type=click.Path(), required=True)
@click.option('--model', '-m', type=click.Choice(['vgg16', 'resnet50v2', 'mobilenetv2']), required=True)
@click.option('--orientation', type=click.Choice(['horizontal', 'vertical', 'horizontal+vertical']), required=True)
@click.option('--output', '-o', type=click.Path(), required=True)
@click.option('--patches', '-p', multiple=True, default=[1], type=int)
@click.option('--width', '-w', type=int, required=True)
def main(colormode, contrast, folds, gpu, height, input, model, orientation, output, patches, width):
    if not os.path.exists(input):
        raise SystemExit('%s does not exist' % input)

    os.makedirs(output, exist_ok=True)
    list_folders = [p for p in pathlib.Path(input).glob('*') if p.is_dir()]

    if len(list_folders) <= 0:
        raise SystemExit('list is emtpy')

    folds = len(list_folders)
    folds = list(range(1, folds + 1))
    patches = list(patches)
    total_samples = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    spec_height = height
    spec_width = width

    print('Feature Extraction Parameters')
    print('Pre-trained model: %s' % model)
    print('Non-overlapping patches per image: %s' % str(patches))
    print('Folds: %s' % str(folds))
    print('Image Dimensions h=%s, w=%s ' % (height, width))
    print('Format string for input: %s ' % input)
    print('Format string for output: %s ' % output)
    print('GPU ID: %d' % gpu)

    for n_patches in patches:
        print('Slicing images into %d non-overlapping patches...' % n_patches)
        tf.keras.backend.clear_session()

        input_shape = get_input_shape(colormode, n_patches, orientation, spec_height, spec_width)
        model, preprocess_input = get_model(model, weights='imagenet', include_top=False, input_shape=input_shape,
                                            pooling='avg')

        for fold in folds:
            input_path_proto = os.path.join(input, 'f%d' % fold)
            print('Extracting features for fold %d...' % fold)

            features = []
            for file in pathlib.Path(input_path_proto).rglob('*'):
                print('file name: %s' % file.name)
                image_sliced = []
                image = tf.keras.preprocessing.image.load_img(file)

                if contrast > 0:
                    image = adjust_contrast(contrast, image)

                spec = tf.keras.preprocessing.image.img_to_array(image)
                extract_features(features, image_sliced, model, n_patches, orientation, preprocess_input, spec)
                save_image(contrast, file, fold, image_sliced, n_patches, output)

            features = np.concatenate(features)
            for extension in ['npy', 'npz']:
                save_file(extension, features, fold, n_patches, output)

            n_samples, n_features = features.shape
            total_samples = total_samples + n_samples

        model_name = model._name
        save_information(colormode, contrast, height, input, model_name, n_features, n_patches, output, total_samples, width)


if __name__ == '__main__':
    main()
