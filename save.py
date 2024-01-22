import numpy as np
import os
import pandas as pd
import pathlib
import shutil
import tensorflow as tf

from typing import LiteralString


def save_image(contrast: float, file: pathlib, fold: int, images: list, patch: int,
               path: LiteralString | pathlib.PurePath | str) -> None:
    dirname = 'image_CONTRAST_%.2f' % contrast if contrast > 0 else 'image'
    path = create_path(path, dirname, 'f%s' % fold)

    for i, image in enumerate(images, start=1):
        filename = '%s_patch=%d+%d%s' % (file.stem, patch, i, file.suffix)
        if patch > 1:
            path_image = create_path(path, file.stem)
            filename = os.path.join(path_image, filename)
        else:
            filename = os.path.join(path, filename)

        tf.keras.preprocessing.image.save_img(filename, image)
        print('%s saved' % filename)


def create_path(path: LiteralString | pathlib.PurePath | str, *args) -> LiteralString | pathlib.PurePath | str:
    path = os.path.join(path, *args)
    os.makedirs(path, exist_ok=True)
    return path


def save_features_npz(fold: int, features: np.ndarray, n_patches: int,
                      path: LiteralString | pathlib.PurePath | str, ) -> None:
    create_path(path, 'features', 'npz')
    filename = 'fold-%d_patches-%d.npz' % (fold, n_patches)
    np.save(filename, features, allow_pickle=True)


def save_features_npy(fold: int, features: np.ndarray, n_patches: int,
                      path: LiteralString | pathlib.PurePath | str, ) -> None:
    create_path(path, 'features', 'npy')
    filename = 'fold-%d_patches-%d.npy' % (fold, n_patches)
    np.save(filename, features, allow_pickle=True)


def save_features_info(input: LiteralString | pathlib.PurePath | str,
                       path: LiteralString | pathlib.PurePath | str) -> None:
    for filename in ['info_dataset.csv', 'info_levels.csv', 'info_samples.csv']:
        if os.path.exists(os.path.join(input, filename)):
            src = os.path.join(input, filename)
            dst = create_path(path, 'features', filename)
            shutil.copy(src, dst)


def save_features(features: np.ndarray,
                  fold: int,
                  input: LiteralString | pathlib.PurePath | str,
                  n_patches: int,
                  path: LiteralString | pathlib.PurePath | str) -> None:
    save_features_npz(fold, features, n_patches, path)
    save_features_npy(fold, features, n_patches, path)
    save_features_info(input, path)


def get_labels(features: np.ndarray, fold: int) -> np.ndarray:
    return np.repeat(fold, features.shape[0])


def save_samples(filenames: list, path: LiteralString | pathlib.PurePath | str) -> None:
    columns = ['filename', 'fold']
    filename = os.path.join(path, 'info_samples2.csv')
    df = pd.DataFrame(filenames, columns=columns, index=None)
    df.to_csv(filename, sep=';', quoting=2, header=True, index=False, lineterminator='\n')


def save_levels(levels: list, path: LiteralString | pathlib.PurePath | str) -> None:
    columns = ['levels', 'count', 'f']
    filename = os.path.join(path, 'info_levels2.csv')
    df = pd.DataFrame(levels, columns=columns, index=None)
    df.to_csv(filename, sep=';', quoting=2, header=True, index=False, lineterminator='\n')


def save_information(color: str,
                     contrast: float,
                     filenames: list,
                     height: int,
                     input: str,
                     levels: list,
                     model: str,
                     n_features: int,
                     n_patches: int,
                     path: LiteralString | pathlib.PurePath | str,
                     total_samples: int, width: int):
    data = {'color': color,
            'contrast': contrast,
            'height': height,
            'input': input,
            'model': model,
            'n_features': n_features,
            'n_patches': n_patches,
            'output': path,
            'total_samples': total_samples,
            'width': width}
    df = pd.DataFrame(data.values(), index=list(data.keys()))
    filename = os.path.join(path, 'features', 'info.csv')
    print('%s saved' % filename)
    df.to_csv(filename, sep=';', quoting=2, header=False, lineterminator='\n')

    save_samples(filenames, path)
    save_levels(levels, path)
