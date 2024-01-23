from typing import Any

import click
import cv2 as cv
import cv2.xfeatures2d
import numpy as np
import os.path
import pandas as pd
import pathlib
import re
import scipy.stats

from PIL import Image, ImageEnhance, ImageOps
from skimage.feature import local_binary_pattern


def lbp(image, label):
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

    return features, features.shape[0] - 1


def surf64(image: Any, label: int):
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=1000)
    kp, histograma = surf.detectAndCompute(image, None)
    print(kp, histograma)
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
    return features, features.shape[0] - 1


def adjust_contrast(contrast: float, filename: str, image, path: str) -> Image.Image:
    enhancer = ImageEnhance.Contrast(image)
    image_contrast = enhancer.enhance(contrast)

    filename = os.path.join(path, filename)
    image_contrast.save(filename)

    return image_contrast


def extract_features(contrast, dataset, extractor, path):
    list_dirs = [p for p in pathlib.Path(path).rglob('*') if p.is_dir()]

    features = []
    n_features = 0
    total_samples = 0
    for d in list_dirs:
        list_images = list(sorted(pathlib.Path(d).glob('*.jpeg')))
        total_samples += len(list_images)

        for i, file in enumerate(list_images, start=1):
            print('[%d/%d] fname: %s' % (i, len(list_images), file.resolve()))
            label = str(d.name).replace('f', '')

            path_im_contrast = str(d).replace(dataset, '%s_CONTRAST_%s' % (dataset, contrast))
            if not os.path.exists(path_im_contrast):
                os.makedirs(path_im_contrast)

            image = Image.open(file.resolve())
            im_contrast = adjust_contrast(contrast, os.path.join(path_im_contrast, file.name), image)
            params = {'image': im_contrast, 'label': label}
            f, n_features = extractor(**params)
            features.append(f)

    path_features = path.replace(dataset, '%s_features_CONTRAST_%s' % (dataset, contrast))

    if not os.path.exists(path_features):
        os.makedirs(path_features)

    fname = '%s.txt' % extractor.__name__
    fname = os.path.join(path_features, fname)
    np.savetxt(fname, np.array(features), fmt='%s')
    print('file %s created' % fname)

    return n_features, path_features, total_samples


def create_df_info(data, path, region=None):
    columns = ['dataset', 'color', 'extractor', 'n_features', 'height', 'level', 'minimum_image', 'input_path',
               'output_path', 'total_samples', 'width', 'contrast']

    if region:
        columns.append('region')

    df = pd.DataFrame(data, columns=columns)
    filename = os.path.join(path, 'info.csv')
    print('file %s created' % filename)
    df.to_csv(filename, header=True, index=False, sep=';', line_terminator='\n', quoting=2)


def create_path(path: str, *args):
    path = os.path.join(path, *args)
    os.makedirs(path, exist_ok=True)
    return path


def save_lbp(features_lbp: np.ndarray, path: str):
    filename = os.path.join(path, 'lbp.txt')
    print('file %s created' % filename)
    np.savetxt(filename, np.array(features_lbp), fmt='%s')


def save_surf(features_surf: np.ndarray, path: str):
    filename = os.path.join(path, 'surf.txt')
    print('file %s created' % filename)
    np.savetxt(filename, np.array(features_surf), fmt='%s')


def save_info(contrast: float, descriptor: str, n_features: int, path: str, total_samples: int, n_patches: int = 1,
              color: str = 'grayscale'):
    # lbp and surf only works grayscale images
    # data = {'n_features': n_features, 'total_samples': , 'contrast': contrast, 'model': descriptor , 'color': color, 'height':, 'width':, 'n_patches': n_patches}
    # df = pd.DataFrame(data.values(), index=list(data.keys()))
    # filename = os.path.join(path, 'info_%s.csv' % descriptor)
    # df.to_csv(filename, header=False, index=True, sep=';', line_terminator='\n', quoting=2)
    pass


def save(contrast: float, features_lbp: np.ndarray, features_surf: np.ndarray, n_features_lbp: int,
         n_features_surf64: int, path: str) -> None:
    path = create_path(path, 'features')
    save_lbp(features_lbp, path)
    save_surf(features_surf, path)
    total_samples = features_surf.shape[0]
    print('total samples: %d' % total_samples)
    save_info(contrast, 'lbp', n_features_surf64, path)


@click.command()
@click.option('--contrast', '-c', type=float, default=0.0)
# @click.option('--descriptor', '-d', type=click.Choice(['lbp', 'surf']), required=True, multiple=True)
@click.option('--input', '-i', type=click.Path(), required=True)
@click.option('--output', '-o', type=click.Path(), default='./non-handcraft')
def main(contrast, input, output):
    if not os.path.exists(input):
        raise IsADirectoryError('input not founded: %s' % output)

    print('loading images from %s' % input)

    features_lbp = []
    features_surf = []
    for file in sorted(pathlib.Path(input).rglob('*[.png,jpg,jpeg]')):
        print(file)
        finds = re.findall(r'/f(\d+)/', str(file))
        if len(finds) == 0:
            raise ValueError('problems in found a label')

        if not finds[0].isnumeric():
            raise ValueError('label is a not numeric')

        image = ImageOps.grayscale(Image.open(file))
        label = int(finds[0])
        if contrast > 0:
            image = adjust_contrast(contrast, file.name, image, output)

        print(image.size)
        image = np.array(image)
        feature, n_features_lbp = lbp(image, label)
        features_lbp.append(feature)

        feature, n_features_surf64 = surf64(image, label)
        features_surf.append(feature)

    save(contrast, features_lbp, features_surf, n_features_lbp, n_features_surf64, output)


if __name__ == '__main__':
    main()
    # print('Loading data...')
    # for contrast in [1.2]:
    #     for dataset in ['regions_dataset']:
    #         for minimum in [5, 10, 20]:
    #             for level in ['specific_epithet_trusted']:
    #                 for image_size in [256, 400, 512]:
    #                     info = []
    #                     for extractor in [lbp, surf64]:
    #                         if dataset == 'regions_dataset':
    #                             regions = []
    #                             for region in ['Norte', 'Nordeste', 'Sul', 'Sudeste', 'Centro-Oeste']:
    #                                 path = os.path.join('/home/xandao/Imagens/', dataset, 'GRAYSCALE', level, region, str(image_size), str(minimum))
    #                                 n_features, output_path, total_samples = extract_features(contrast, dataset, extractor, path)
    #                                 info.append([dataset, 'GRAYSCALE', extractor.__name__, n_features, image_size, level, minimum,
    #                                      path, output_path, total_samples, image_size, contrast, region])
    #                                 create_df_info(info, output_path, region=region)
    #                         else:
    #                             path = os.path.join('/home/xandao/Imagens/', dataset, 'GRAYSCALE', level, str(image_size), str(minimum))
    #                             n_features, output_path, total_samples = extract_features(contrast, dataset, extractor, path)
    #                             info.append([dataset, 'GRAYSCALE', extractor.__name__, n_features, image_size, level, minimum,
    #                                           path, output_path, total_samples, image_size, contrast])
    #                             create_df_info(info, output_path)
