import os
import pathlib
from typing import LiteralString

import numpy as np
import pandas as pd


def save_npy(features: np.ndarray, fold: int, patch: int, output: pathlib.Path | LiteralString | str) -> None:
    """
    Cria uma pasta para salvar as features no formato npy de um determinado fold.
    :param features: matriz com as características extraídas.
    :param fold: a classe que pertence aquelas features.
    :param patch: quantidade de divisões feitas nas imagens.
    :param output: local onde será salvo as features.
    :return: .
    """
    output = os.path.join(output, 'features', 'npy', 'f%d' % fold)
    os.makedirs(output, exist_ok=True)
    filename = 'fold-%d_patches-%d.npy' % (fold, patch)
    filename = os.path.join(output, filename)
    np.save(filename, features, allow_pickle=True)


def save_npz(features: np.ndarray, fold: int, patch: int, output: pathlib.Path | LiteralString | str) -> None:
    """
    Cria uma pasta para salvar as features no formato npz de um determinado fold.
    :param features: matriz com as características extraídas.
    :param fold: a classe que pertence aquelas features.
    :param patch: quantidade de divisões feitas nas imagens.
    :param output: local onde será salvo as features.
    :return: .
    """
    output = os.path.join(output, 'features', 'npz', 'f%d' % fold)
    os.makedirs(output, exist_ok=True)
    filename = 'fold-%d_patches-%d.npz' % (fold, patch)
    filename = os.path.join(output, filename)
    np.savez_compressed(filename, x=features, y=np.repeat(fold, features.shape[0]))


def save_features(fold: int, features, format: str, patch: int, output: pathlib.Path | LiteralString | str) -> None:
    """
    Chama a função para salvar as features, baseado na opção que foi escolhida pelo usuário.
    :param fold: a classe que pertence aquelas features.
    :param features: matriz com as características extraídas.
    :param format: formato escolhido pelo usuário.
    :param patch: quantidade de divisões feitas nas imagens.
    :param output: local onde será salvo as features.
    """
    match format:
        case 'npy':
            save_npy(features, fold, patch, output)
        case 'npz':
            save_npz(features, fold, patch, output)
        case 'all':
            save_npy(features, fold, patch, output)
            save_npz(features, fold, patch, output)


def save_images(fold: int, images: list, output: pathlib.Path | LiteralString | str) -> None:
    """
    Cria uma pasta para salvar as imagens que foram divididas.
    :param fold: a classe que pertence aquelas imagens.
    :param images: lista com as imagens que deverão ser salvas.
    :param output: local onde será salvo as imagens.
    """
    if len(images) <= 0:
        raise ValueError('No images found')

    p = os.path.join(output, 'images', 'f%d' % fold)
    os.makedirs(p, exist_ok=True)

    for image in images:
        image.save_patches(p)


def save_csv(fold: int, features: np.ndarray, images: list, patch: int, output: pathlib.Path | LiteralString | str):
    save_dataset(features, fold, images, output, patch)
    save_samples(images, output)


def save_dataset(features, fold, images: list, output, patch):
    n_samples = features.shape[0]
    n_features = features.shape[1]
    data = {
        'fold': [fold],
        'patches': [patch],
        'features': [n_features],
        'samples': [len(images)],
        'samples+patch': np.sum([len(image.patches) for image in images]),
    }
    df = pd.DataFrame(data, columns=list(data.keys()))
    filename = os.path.join(output, 'dataset.csv')
    df.to_csv(filename, sep=';', quoting=2, quotechar='"', encoding='utf-8', index=False, header=True)


def save_samples(images, output):
    data = {'filename': [image.filename for image in images], 'fold': [image.fold for image in images]}
    df = pd.DataFrame(data, columns=list(data.keys()))
    filename = os.path.join(output, 'samples.csv')
    df.to_csv(filename, sep=';', quoting=2, quotechar='"', encoding='utf-8', index=False, header=True)


def save(fold: int, features: np.ndarray, format: str, images: list, patch: int,
         output: pathlib.Path | LiteralString | str) -> None:
    """
    Chama as funções que salvam as features, as imagens e as informações da extração.
    :param fold: a classe que pertence aquelas imagens.
    :param features: matriz com as características extraídas.
    :param format: formato escolhido pelo usuário.
    :param images: lista com as imagens que deverão ser salvas.
    :param patch: quantidade de divisões feitas nas imagens.
    :param output:  local onde serão salvos as imagens e as features.
    """
    save_features(fold, features, format, patch, output)
    save_images(fold, images, output)
    save_csv(fold, features, images, patch, output)
