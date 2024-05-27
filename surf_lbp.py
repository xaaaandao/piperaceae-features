
import click
import cv2 as cv
import cv2.xfeatures2d
import numpy as np
import os.path
import pandas as pd
import pathlib
import PIL
import scipy.stats
from skimage.feature import local_binary_pattern
import tensorflow as tf
from typing import LiteralString

from image import Image, adjust_contrast
from save import save


def lbp(image: PIL.Image, label: int) -> np.ndarray:
    """
    Extrai features usando o algoritmo SURF.
    :param image: imagem que será extraída as features.
    :param label: classe que pertence aquela imagem.
    :return: np.ndarray: matriz com as features extraídas da imagem.
    """
    n_neighbors = 8
    radius = 1
    n_points = 8 * radius

    n_bins = n_neighbors * (n_neighbors - 1) + 3
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')

    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-6)

    label = np.array([label], dtype=int)
    features = np.append(hist, label)

    return features


def surf64(image: PIL.Image, label: int) -> np.ndarray:
    """
    Extrai features usando o algoritmo SURF.
    :param image: imagem que será extraída as features.
    :param label: classe que pertence aquela imagem.
    :return: np.ndarray: matriz com as features extraídas da imagem.
    """
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=1000)
    kp, histograma = surf.detectAndCompute(image, None)

    if not len(histograma.shape) == 2:
        raise SystemError('histograma SURF error')

    v_hist = histograma.shape[0]

    vetor_aux = np.mean(histograma, axis=0)
    mean = vetor_aux[0:vetor_aux.shape[0]]

    vetor_aux = np.std(histograma, axis=0)
    desv_pad = vetor_aux[0:vetor_aux.shape[0]]

    vetor_aux = scipy.stats.kurtosis(histograma, bias=False, axis=0)
    kurtosis = vetor_aux[0:vetor_aux.shape[0]]

    vetor_aux = scipy.stats.skew(histograma, bias=False, axis=0)
    skew = vetor_aux[0:vetor_aux.shape[0]]

    v_hist = np.array([v_hist], dtype=int)
    label = np.array([label], dtype=int)

    features = np.concatenate((v_hist, mean, desv_pad, kurtosis, skew, label))

    # -1 to label
    return features


def extract_features(contrast: float,
                     descriptor: str,
                     folds: int,
                     format: list,
                     gpuid: int,
                     height: int,
                     input: pathlib.Path | LiteralString | str,
                     orientation: str,
                     output: pathlib.Path | LiteralString | str,
                     patches: int,
                     save_images: bool,
                     width: int):
    """
    Extrai as features das imagens presentes no diretório passado por parâmetro.
    :param contrast: valor do contraste a ser aplicado na imagem.
    :param descriptor: nome do descritor a ser utilizado.
    :param folds: número de folds (ou número de classes).
    :param gpuid: número da GPU.
    :param height: altura da imagem.
    :param input: diretório de entrada das imagens.
    :param model: modelo que será utilizado para extrair as features.
    :param orientation: orientação da divisão da imagem (horizontal, vertical ou ambas as direções).
    :param output: diretóiro de saída.
    :param patches: número de patches (divisões) na imagem/.
    :param width: largura da imagem.
    :return:
    """
    features = []
    images = []
    for image in list(sorted(pathlib.Path(input).rglob('*.jpeg'))):
        im = PIL.Image.open(image.resolve())

        if contrast > 0:
            im = np.array(adjust_contrast(contrast, im))

        img = Image(image, list(im))

        if save_images:
            img.save_patches(output)

        images.append(img)
        match descriptor:
            case 'lbp':
                features.append(lbp(im, img.fold))
            case 'surf':
                features.append(surf64(im, img.fold))

    save(descriptor, features, images, output)


@click.command()
@click.option('-c', '--contrast', type=float, default=0.0)
@click.option('-d', '--descriptor', type=click.Choice(['surf', 'lbp']), required=True)
@click.option('--formats', type=click.Choice(['all', 'npy', 'npz']),
              required=True,
              help='all: create features file in two format, npy: create features in npy format and npz: create features in npz format;')
# @click.option('-f', '--folds', type=int)
# @click.option('--gpuid', type=int, default=0)
# @click.option('-h', '--height', type=int, required=True)
@click.option('-i', '--input', required=True)
# @click.option('--orientation', type=click.Choice(['horizontal', 'vertical', 'horizontal+vertical']), required=True)
@click.option('-o', '--output', default='output')
# @click.option('-p', '--patches', required=True, default=[1], multiple=True)
@click.option('-s', '--save_images', is_flag=True)
# @click.option('-w', '--width', type=int, required=True)
def main(contrast: float, formats: list, folds: int, gpuid: int, height: int, input, model, orientation, output,
         patches: int, save_images: bool, width: int):
    print('Feature Extraction Parameters')
    print('Pre-trained model: %s' % model)
    print('Non-overlapping patches per image: %s' % str(patches))
    print('Folds: %s' % str(folds))
    print('Image Dimensions h=%s, w=%s ' % (height, width))
    print('Format string for input: %s ' % input)
    print('Format string for exemplos: %s ' % output)
    print('GPU ID: %d' % gpuid)

    extract_features(contrast, folds, formats, gpuid, height, input, model, orientation, output, patches, save_images, width)


if __name__ == '__main__':
    main()
