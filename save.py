import os
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf


def save_image(contrast: float, file, fold: int, image_sliced: list, patch: int, output):
    fold = 'f%s' % fold
    output_to_imgs = os.path.join(output, 'images', fold)

    if contrast > 0:
        output_to_imgs = os.path.join(output, 'images+contrast=%f' % contrast, fold)

    os.makedirs(output_to_imgs, exist_ok=True)
    for i, image in enumerate(image_sliced, start=1):
        fname = '%s_patch=%d+%d%s' % (file.stem, patch, i, file.suffix)
        if patch > 1:
            os.makedirs(os.path.join(output_to_imgs, file.stem), exist_ok=True)
            fname = os.path.join(output_to_imgs, file.stem, fname)
        else:
            fname = os.path.join(output_to_imgs, fname)
        tf.keras.preprocessing.image.save_img(fname, image)
        print('%s saved' % fname)


def save_features(extension: str, features, fold: int, input:str, n_patches: int, output: str):
    out = os.path.join(output, 'features', extension)
    os.makedirs(out, exist_ok=True)

    filename = 'fold-%d_patches-%d.%s' % (fold, n_patches, extension)
    filename = os.path.join(out, filename)

    print('%s save' % filename)
    if extension == 'npy':
        np.save(filename, features, allow_pickle=True)
    else:
        np.savez_compressed(filename, x=features, y=get_classes(features, fold))

    for f in ['info_dataset.csv', 'info_levels.csv', 'info_samples.csv']:
        if os.path.exists(os.path.join(input, f)):
            shutil.copy(os.path.join(input, f), os.path.join(output, 'features'))


def get_classes(features, fold: int):
    return np.repeat(fold, features.shape[0])


def save_information(colormode: str, contrast: float, height: int, input: str, model: str, n_features: int, n_patches: int, output,
                     total_samples: int, width: int):
    data = {'colormode': colormode,
            'contrast': contrast,
            'height': height,
            'input': input,
            'model': model,
            'n_features': n_features,
            'n_patches': n_patches,
            'output': output,
            'total_samples': total_samples,
            'width': width}
    df = pd.DataFrame(data.values(), index=list(data.keys()))
    filename = os.path.join(output, 'features', 'info.csv')

    print('%s saved' % filename)
    df.to_csv(filename, sep=';', quoting=2, header=False, lineterminator='\n')
