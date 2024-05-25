from typing import LiteralString

import click
import glob
import numpy as np
import os
import pathlib
import tensorflow as tf

from image import adjust_contrast
from model import get_model, get_input_shape
from patch import next_patch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def save_features():
    pass


def save_images():
    pass


def extract_features(contrast: float, folds: int, gpuid: int, height: int, input: pathlib.Path | LiteralString | str,
                     model: str, orientation: str,
                     output: pathlib.Path | LiteralString | str, patches: int, width: int):
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
    input_path_proto = os.path.join(input, 'f%d', '*.jpeg')

    for patch in patches:
        print('Slicing images into %d non-overlapping patches...' % (patch))
        tf.keras.backend.clear_session()

        input_shape = get_input_shape(orientation, patch, height, width)

        model, preprocess_input = get_model(model, weights='imagenet', include_top=False,
                                            input_shape=input_shape, pooling='avg')
        total_samples = 0
        n_features = 0
        for fold in folds:
            print('Extracting features for fold %d...' % (fold))
            if len(glob.glob(input_path_proto % (fold))) == 0:
                raise RuntimeError('No files found in: %s' % (input_path_proto % (fold)))

            features = []
            for fname in sorted(glob.glob(input_path_proto % (fold))):
                images = []
                im = tf.keras.preprocessing.image.load_img(fname)

                if contrast > 0:
                    im = adjust_contrast(contrast, im)

                spec = tf.keras.preprocessing.image.img_to_array(im)
                for p in next_patch(spec, patch):
                    p = preprocess_input(p)

                    # Armazena na lista a imagem recortada
                    images.append(tf.keras.preprocessing.image.array_to_img(p))
                    p = np.expand_dims(p, axis=0)

                    # Armazena na lista as features extraídas
                    features.append(model.predict(p))

                save_images()

            features = np.concatenate(features)

            save_features()


@click
@click.option('-c', '--contrast', type=float)
@click.option('-f', '--format')
@click.option('--gpuid', type=int, default=0)
@click.option('-h', '--height', type=int, required=True)
@click.option('-i', '--input', type=click.Path, required=True)
@click.option('-m', '--model', required=True)
@click.option('--orientation', type=click.Choice(['horizontal', 'vertical', 'horizontal+vertical']),
              default='horizontal')
@click.option('--output')
@click.option('-p', '--patch', required=True, default=[1], multiple=True)
@click.option('-s', '--save_images', type=bool)
@click.option('-w', '--width', type=int, required=True)
def main(folds: int, gpuid: int, height: int, input, model, orientation, output, patch: int, save_images: bool,
         width: int):
    print('Feature Extraction Parameters')
    print('Pre-trained model: %s' % model)
    print('Non-overlapping patches per image: %s' % str(patch))
    print('Folds: %s' % str(folds))
    print('Image Dimensions h=%s, w=%s ' % (height, width))
    print('Format string for input: %s ' % input)
    print('Format string for output: %s ' % output)
    print('GPU ID: %d' % gpuid)

    folders = [p for p in pathlib.Path(input).glob('*') if p.is_dir()]
    folds = len(folders)
    folds = list(range(1, folds + 1))


if __name__ == '__main__':
    main()
