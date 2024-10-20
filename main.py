import click
import datetime
import numpy as np
import os
import pathlib
import tensorflow as tf

from typing import LiteralString

from image import adjust_contrast, Image
from model import get_model, get_input_shape
from patch import next_patch
from save import save, save_csv, save_patches

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

dt_now = datetime.datetime.now()
dt_now = dt_now.strftime('%d-%m-%Y-%H-%M-%S')


def extract_features(color: str,
                     contrast: float,
                     folds: int,
                     format: list,
                     gpuid: int,
                     height: int,
                     input: pathlib.Path | LiteralString | str,
                     minimum: int,
                     model: str,
                     name: str,
                     orientation: str,
                     output: pathlib.Path | LiteralString | str,
                     patches: int,
                     regions: str,
                     save_images: bool,
                     width: int):
    """
    Para cada valor de pacthes é feito o corte na imagem, por exemplo: patch=3 serão feitas três divisões nas imagens.
    Logo após, é carregado o modelo que foi definido pelo usuário. No final, é utilizado o modelo para extrair as
    características de cada imagem que estão disponíveis em cada fold.
    :param contrast: valor do contraste a ser aplicado na imagem.
    :param folds: número de folds (ou número de classes).
    :param gpuid: número da GPU.
    :param height: altura da imagem.
    :param input: diretório de entrada das imagens.
    :param model: modelo que será utilizado para extrair as features.
    :param orientation: orientação da divisão da imagem (horizontal, vertical ou ambas as direções).
    :param output: diretóiro de saída.
    :param patches: número de patches (divisões) na imagem/.
    :param width: largura da imagem.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)
    os.makedirs(output, exist_ok=True)

    images = []
    for patch in patches:
        print('Slicing patch_images into %d non-overlapping patches...' % (patch))
        tf.keras.backend.clear_session()

        input_shape = get_input_shape(orientation, patch, height, width)
        model_name = model
        model, preprocess_input = get_model(model, weights='imagenet', include_top=False,
                                            input_shape=input_shape, pooling='avg')
        n_features = 0
        last_idx = 0
        for idx, fold in enumerate(sorted(pathlib.Path(input).glob('*')), start=1):
            if fold.is_dir():
                features = []
                for fname in sorted(pathlib.Path(fold).glob('*.jpeg')):
                    patch_images = []
                    image = tf.keras.preprocessing.image.load_img(fname)

                    if contrast > 0:
                        image = adjust_contrast(contrast, image)

                    spec = tf.keras.preprocessing.image.img_to_array(image)
                    for p in next_patch(spec, patch, orientation):
                        p = preprocess_input(p)

                        # Armazena na lista a imagem recortada
                        if save_images:
                            patch_images.append(tf.keras.preprocessing.image.array_to_img(p))
                        p = np.expand_dims(p, axis=0)

                        # Armazena na lista as features extraídas
                        features.append(model.predict(p))

                    i = Image(fname, patch_images)
                    images.append(i)
                features = np.concatenate(features)
                save(features, idx, format, model_name, output, patch)
                n_features = features.shape[1]
                last_idx = idx
        save_csv(color, contrast, last_idx, format, height, images, input, minimum, model_name, name, n_features, output, patch, regions, width)
    save_patches(images, output)


@click.command()
@click.option('-C', '--color', type=str, default='RGB')
@click.option('-c', '--contrast', type=float, default=0.0)
@click.option('--formats', type=click.Choice(['all', 'npy', 'npz']),
              required=True,
              help='all: create features file in two format, npy: create features in npy format and npz: create features in npz format;')
@click.option('-f', '--folds', type=int)
@click.option('--gpuid', type=int, default=0)
@click.option('-h', '--height', type=int, required=True)
@click.option('-i', '--input', required=True)
@click.option('-m', '--model', type=click.Choice(['mobilenetv2', 'vgg16', 'resnet50v2']), required=True)
@click.option('--minimum', type=int, required=False)
@click.option('-m', '--model', type=click.Choice(['mobilenetv2', 'vgg16', 'resnet50v2']), required=True)
@click.option('-n', '--name', type=str, required=False)
@click.option('--orientation', type=click.Choice(['horizontal', 'vertical', 'horizontal+vertical']), required=True)
@click.option('-o', '--output', default='output')
@click.option('-p', '--patches', required=True, default=[1], multiple=True)
@click.option('-r', '--regions', required=False, type=str)
@click.option('-s', '--save_images', is_flag=True)
@click.option('-w', '--width', type=int, required=True)
def main(color: str, contrast: float, formats: list, folds: int, gpuid: int, height: int, input, minimum, model, name,
         orientation, output,
         patches: int, regions:str, save_images: bool, width: int):
    if 'GRAYSCALE'.lower() in input.lower():
        color = 'GRAYSCALE'

    print('Feature Extraction Parameters')
    print('Pre-trained model: %s' % model)
    print('Non-overlapping patches per image: %s' % str(patches))
    print('Folds: %s' % str(folds))
    print('Image Dimensions h=%s, w=%s ' % (height, width))
    print('Format string for input: %s ' % input)
    print('Format string for exemplos: %s ' % output)
    print('GPU ID: %d' % gpuid)

    patches = list(patches)
    extract_features(color, contrast, folds, formats, gpuid, height, input, minimum, model, name, orientation, output,
                     patches, regions,
                     save_images, width)


if __name__ == '__main__':
    main()
