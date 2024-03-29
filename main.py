import click
import numpy as np
import os
import pathlib
import tensorflow as tf

from cnn import get_model, extract_features
from image import adjust_contrast, get_input_shape
from save import save_image, save_features, save_information

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


@click.command()
@click.option('--color', type=click.Choice(['RGB', 'grayscale']), default='RGB', help='Color of image input.')
@click.option('--contrast', type=float, required=False, default=0, help=' Value that to use in adjusting the contrast.')
@click.option('--folds', '-f', type=int, default=3, help='Number of classes in the dataset.')
@click.option('--gpu', type=int, default=0, help='ID of GPU.')
@click.option('--height', '-h', type=int, required=True, help='Height of image input.')
@click.option('--input', '-i', type=click.Path(), required=True, help='Path to images.')
@click.option('--model', '-m', type=click.Choice(['vgg16', 'resnet50v2', 'mobilenetv2']), required=True, help='A model to extract the features.')
@click.option('--orientation', type=click.Choice(['horizontal', 'vertical', 'horizontal+vertical']), required=True, help='An orientation of patches must made in an image.')
@click.option('--output', '-o', type=click.Path(), required=True, help='Path to store the NPZ, NPY, and CSV files.')
@click.option('--patches', '-p', multiple=True, default=[1], type=int, help='A number the patches must made in an image.')
@click.option('--width', '-w', type=int, required=True, help='Width of image input.')
def main(color, contrast, folds, gpu, height, input, model, orientation, output, patches, width):
    if not os.path.exists(input):
        raise SystemExit('%s does not exist' % input)

    os.makedirs(output, exist_ok=True)

    folds = list(range(1, folds + 1))
    patches = list(patches)
    total_samples = 0
    n_features = 0
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

        input_shape = get_input_shape(color, n_patches, orientation, spec_height, spec_width)
        model, preprocess_input = get_model(model, weights='imagenet', include_top=False, input_shape=input_shape,
                                            pooling='avg')

        levels = []
        filenames = []
        model_name = model._name
        for fold in folds:
            input_path_proto = os.path.join(input, 'f%d' % fold)
            print('Extracting features for fold %d...' % fold)

            features = []
            for file in pathlib.Path(input_path_proto).rglob('*'):
                print('filename: %s' % file.name)
                images = []
                image = tf.keras.preprocessing.image.load_img(file)

                if contrast > 0:
                    image = adjust_contrast(contrast, image)

                spec = tf.keras.preprocessing.image.img_to_array(image)
                extract_features(features, images, model, n_patches, orientation, preprocess_input, spec)
                save_image(contrast,file, fold, image, images, model_name, n_patches, output)
                filenames.append([file.name, fold])

            features = np.concatenate(features)
            save_features(features, fold, input, model_name, n_patches, output)

            n_samples, n_features = features.shape
            total_samples = total_samples + n_samples
            n_features = n_features + 0
            levels.append([input_path_proto, len(list(pathlib.Path(input_path_proto).rglob('*'))), fold])

        save_information(color, contrast, filenames, height, input, levels, model_name, n_features, n_patches, output, total_samples, width)


if __name__ == '__main__':
    main()
