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
# from typing import LiteralString

from image import Image
# from save import save


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
    surf = cv2.xfeatures2d.SURF_create()
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

    if np.isnan(features).any():
        raise ValueError('contains nan')
    
    # -1 to label
    return features


# model = descriptor
def save_dataset(descriptor, features, height, images, minimum, mode, name, output, width, format='txt', patch=1, regions=None):
    data = {
        'color': [mode],
        'fold': [np.max(features[:, -1])],
        'format': [format],
        'height': [height],
        'patch': [patch],
        'count_features': [features[0].shape[0]-1],
        'count_samples': [len(images)],
        'model': [descriptor],
        'name': [name],
        'minimum': [minimum],
        'regions': [regions],
        'count_samples+patch': [len(images)/patch],
        'width': [width],
    }
    df = pd.DataFrame(data, columns=list(data.keys()))
    filename = os.path.join(output, 'dataset.csv')
    df.to_csv(filename, sep=';', quoting=2, quotechar='"', encoding='utf-8', index=False, header=True)
    print(f'saving {filename}')

def save_samples(descriptor, images, output):
    data = {'filename': [image.filename for image in sorted(images, key=lambda x: x.filename)],
            'fold': [image.fold for image in sorted(images, key=lambda x: x.filename)],
            'specific_epithet': [image.fold for image in sorted(images, key=lambda x: x.filename)]}
    df = pd.DataFrame(data, columns=list(data.keys()))
    filename = os.path.join(output, 'samples.csv')
    df.to_csv(filename, sep=';', quoting=2, quotechar='"', encoding='utf-8', index=False, header=True)

def save_features(descriptor, features, output):
    filename = os.path.join(output, '%s.txt' % descriptor)
    np.savetxt(filename, features)
    print(f'saving {filename}')

def save(descriptor, features, height, images, minimum, mode, name, output, width):
    output = os.path.join(output, descriptor)
    os.makedirs(output, exist_ok=True)
    save_dataset(descriptor, features, height, images, minimum, mode, name, output, width)
    save_samples(descriptor, images, output)
    save_features(descriptor, features, output)

def extract_features(descriptor, input, minimum, name, output):                    
    """
    Extrai as features das imagens presentes no diretório passado por parâmetro.
    :param descriptor: nome do descritor a ser utilizado.
    :param input: diretório de entrada das imagens.
    :param output: diretóiro de saída.
    :return:
    """
    features = []
    images = []
    
    for p in list(sorted(pathlib.Path(input).rglob('*.jpeg'))):
        print(p)
        image = PIL.Image.open(p.resolve())
        height = image.height
        width = image.width
        mode = 'GRAYSCALE' if 'L' in image.mode else 'RGB'
        image = np.asarray(image)
        
        fold = p.parent.name.replace('f', '')
        try:
            fold.isnumeric()
        except:
            raise ValueError

        i = Image(str(p), fold, p.parent.name)
        images.append(i)

        match descriptor:
            case 'lbp':
                features.append(lbp(image, i.fold))
            case 'surf':
                features.append(surf64(image, i.fold))

    features = np.array(features)
    save(descriptor, features, height, images, minimum, mode, name, output, width)

def main():
    # print(input)
    for m in [5, 10, 20]:
        for s in [256, 400, 512]:
            input = f'/home/xandao/Documentos/pr_dataset+{m}/GRAYSCALE/{s}/original'
            output = f'/home/xandao/Documentos/pr_dataset+{m}/GRAYSCALE/{s}/'
            extract_features('lbp', input, m, 'pr_dataset', output)
            extract_features('surf', input, m, 'pr_dataset', output)


if __name__ == '__main__':
    main()
