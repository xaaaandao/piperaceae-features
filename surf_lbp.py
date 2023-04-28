import cv2 as cv
import cv2.xfeatures2d
import numpy as np
import os.path
import pandas as pd
import pathlib
import scipy.stats

from PIL import Image, ImageEnhance
from skimage.feature import local_binary_pattern


def lbp(image, label):
    n_neighbors = 8
    radius = 1
    n_points = 8 * radius

    n_bins = n_neighbors * (n_neighbors - 1) + 3
    lbp = local_binary_pattern(image, n_points, radius)

    hist, _ = np.histogram(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins))

    label = np.array([label], dtype=int)
    features = np.append(hist, label)

    print(features.shape)
    return features, features.shape[0] - 1


def surf(image, label):
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=1000, nOctaves=5, nOctaveLayers=3)
    kp, histograma = surf.detectAndCompute(image, None)

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

    print(features.shape)
    # -1 to label
    return features, features.shape[0] - 1


def adjust_contrast(image):
    # image brightness enhancer
    enhancer = ImageEnhance.Contrast(image)
    factor = 1.5
    image = enhancer.enhance(factor)
    return np.array(image)


def extract_features(dataset, path):
    list_dirs = [p for p in pathlib.Path(path).rglob('*') if p.is_dir()]

    features = []
    n_features = 0
    total_samples = 0
    for d in list_dirs:
        list_images = pathlib.Path(d).glob('*.jpeg')
        total_samples += len(list(list_images))
        for i, file in enumerate(list_images, start=1):
            print('[%d/%d] fname: %s' % (i, len(list(list_images)), file.resolve()))
            label = str(d.name).replace('f', '')
            image = Image.open(file.resolve())
            image = adjust_contrast(image)
            f, n_features = lbp(image, label)
            features.append(f)

    path_final = path.replace(dataset, '%s_features' % dataset)

    if not os.path.exists(path_final):
        os.makedirs(path_final)

    # fname = '%s.txt' % (extractor.__name__)
    fname = os.path.join(path_final, 'lbp.txt')
    np.savetxt(fname, np.array(features), fmt='%s')
    print('file %s created' % fname)

    return n_features, path_final, total_samples


def create_df_info(data, path):
    columns = ['dataset', 'color', 'extractor', 'n_features', 'height', 'level', 'minimum_image', 'input_path',
               'output_path', 'total_samples', 'width']

    df = pd.DataFrame(data, columns=columns)
    filename = os.path.join(path, 'info.csv')
    print('file %s created' % filename)
    df.to_csv(filename, header=True, index=False, sep=';', line_terminator='\n')


for dataset in ['pr_dataset']:  # , 'br_dataset', 'regions_dataset']:
    for minimum in [5]:  # , 10, 20]:
        for level in ['specific_epithet_trusted']:
            for image_size in [512]:  # , 400, 256]:
                if dataset == 'regions_dataset':
                    regions = []
                    for region in ['Norte', 'Nordeste', 'Sul', 'Sudeste', 'Centro-Oeste']:
                        path = os.path.join('/home/xandao/Imagens/', dataset, 'GRAYSCALE', str(image_size),
                                            level, region, str(minimum))
                        # list_n_features, list_total_samples, output_paths = extract_features(dataset, path)
                        # info = [dataset, 'GRAYSCALE', path, image_size]
                        # regions.append(region)
                        # create_df_info(info, path)
                else:
                    path = os.path.join('/home/xandao/Imagens/', dataset, 'GRAYSCALE', level, str(image_size),
                                        str(minimum))
                    n_features, output_path, total_samples = extract_features(dataset, path)
                    # info.append([dataset, 'GRAYSCALE', extractor.__name__, n_features, image_size, level, minimum,
                    #              path, output_path, total_samples, image_size])
                    # create_df_info(info, output_path)
