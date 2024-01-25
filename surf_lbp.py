import collections

import click
import cv2 as cv
import cv2.xfeatures2d
import datetime
import numpy as np
import os.path
import pandas as pd
import pathlib
import re
import scipy.stats

from PIL import Image, ImageEnhance, ImageOps
from skimage.feature import local_binary_pattern
from typing import Any

datefmt = '%d-%m-%Y+%H-%M-%S'
dateandtime = datetime.datetime.now().strftime(datefmt)


def lbp(image: Image, label: int, n_neighbors: int = 8, radius: int = 1):
    n_points = 8 * radius

    n_bins = n_neighbors * (n_neighbors - 1) + 3
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')

    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-6)

    label = np.array([label], dtype=int)
    features = np.append(hist, label)

    return features


def surf64(image: Any, label: int):
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

    return features


def adjust_contrast(contrast: float, file: pathlib, image, path) -> Image.Image:
    enhancer = ImageEnhance.Contrast(image)
    image_contrast = enhancer.enhance(contrast)

    path = os.path.join(path, 'images', 'pillow', str(file.parent.name))
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, str(file.name))
    image_contrast.save(filename)

    return image_contrast


def create_path(path: str, *args):
    path = os.path.join(path, *args)
    os.makedirs(path, exist_ok=True)
    return path


def save_lbp(features_lbp: np.ndarray, path: str):
    path = create_path(path, 'features', 'lbp')
    filename = os.path.join(path, 'lbp.txt')
    print('file %s created' % filename)
    np.savetxt(filename, np.array(features_lbp), fmt='%s')


def save_surf(features_surf: np.ndarray, path: str):
    path = create_path(path, 'features', 'surf')
    filename = os.path.join(path, 'surf.txt')
    print('file %s created' % filename)
    np.savetxt(filename, np.array(features_surf), fmt='%s')


def save_info(contrast: float, descriptor: str, height: int, n_features: int, path: str, total_samples: int, width: int,
              n_patches: int = 1,
              color: str = 'grayscale'):
    # lbp and surf only works grayscale images
    data = {'n_features': n_features - 1, 'total_samples': total_samples, 'contrast': contrast,
            'descriptor': descriptor,
            'color': color, 'height': height, 'width': width, 'n_patches': n_patches}
    df = pd.DataFrame(data.values(), index=list(data.keys()))
    filename = os.path.join(path, 'features', descriptor, 'info.csv')
    print('Saving %s' % filename)
    df.to_csv(filename, sep=';', quoting=2, header=False, index=True, lineterminator='\n')


def save_samples(descriptor: str, path, samples: list):
    df = pd.DataFrame(samples, columns=['filename', 'label'])
    filename = os.path.join(path, 'features', descriptor, 'info_samples.csv')
    print('Saving %s' % filename)
    df.to_csv(filename, sep=';', quoting=2, header=True, index=False, lineterminator='\n')


def save_levels(descriptor: str, levels: list, path):
    levels = [[k[0], k[1], v] for k, v in dict(collections.Counter(levels)).items()]
    df = pd.DataFrame(levels, columns=['levels', 'f', 'count'])
    filename = os.path.join(path, 'features', descriptor, 'info_levels.csv')
    print('Saving %s' % filename)
    df.to_csv(filename, sep=';', quoting=2, header=True, index=False, lineterminator='\n')


def save(contrast: float, features_lbp: np.ndarray, features_surf: np.ndarray, height: int, levels: list, path: str,
         samples: list, width: int) -> None:
    path = create_path(path)
    save_lbp(features_lbp, path)
    save_surf(features_surf, path)
    save_info(contrast, 'surf', height, features_surf.shape[1], path, features_surf.shape[0], width)
    save_info(contrast, 'lbp', height, features_lbp.shape[1], path, features_lbp.shape[0], width)
    for descriptor in ['surf', 'lbp']:
        save_samples(descriptor, path, samples)
        save_levels(descriptor, levels, path)


@click.command()
@click.option('--contrast', '-c', type=float, default=0.0, help='Value that to use in adjusting the contrast.')
@click.option('--input', '-i', type=click.Path(), required=True, help='Path to images.')
@click.option('--output', '-o', type=click.Path(), default='./non-handcraft', help='Path to store the text and CSV files.')
def main(contrast, input, output):
    if not os.path.exists(input):
        raise IsADirectoryError('input not founded: %s' % output)

    print('loading images from %s' % input)

    features_lbp = []
    features_surf = []
    levels = []
    samples = []
    height, width = (0, 0)
    for file in sorted(pathlib.Path(input).rglob('*[.png,jpg,jpeg]')):
        print(file)
        finds = re.findall(r'/f(\d+)/', str(file))
        if len(finds) == 0:
            raise ValueError('problems in found a label')

        if not finds[0].isnumeric():
            raise ValueError('label is a not numeric')

        image = ImageOps.grayscale(Image.open(file))
        height, width = image.size
        label = int(finds[0])
        if contrast > 0:
            image = adjust_contrast(contrast, file, image, output)

        image = np.array(image)
        levels.append((label, label))
        samples.append([file.name, label])
        features_lbp.append(lbp(image, label))
        features_surf.append(surf64(image, label))

    save(contrast, np.array(features_lbp), np.array(features_surf), height, levels, output, samples, width)


if __name__ == '__main__':
    main()
