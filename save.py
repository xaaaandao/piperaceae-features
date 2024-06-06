import os
import pathlib
from typing import LiteralString, overload

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


def save_features(features, fold: int, format: str, patch: int, output: pathlib.Path | LiteralString | str) -> None:
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


# def save_features(descriptor: str, features: np.ndarray, output):
#     p = os.path.join(output, descriptor)
#     os.makedirs(p, exist_ok=True)
#
#     filename = '%s.txt' % descriptor
#     filename = os.path.join(p, filename)
#     np.savetxt(filename, np.array(features), fmt='%s')


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

# save_csv(color, contrast, fold, height, images, input, minimum, model, name, n_features, output, patch, width)
def save_csv(color:str, contrast: str,
             fold: int,
             format: int,
             height: int,
             images: list,
             input:  pathlib.Path | LiteralString | str,
             minimum: int, model: str,
             name: str,
             n_features: int,
             output: pathlib.Path | LiteralString | str, patch: int,
             width: int):
    """
    Gera dois arquivos CSV:
    1- Salva as informações do dataset.
    2- Salva as informações das amostras que foram extraídas as características.
    :param fold: a classe que pertence aquelas imagens.
    :param features: quantidade de features extraídas.
    :param images: lista com as imagens que deverão ser salvas.
    :param model: rede usada para extrair as características.
    :param output: local onde será salvo as imagens.
    :param patch: quantidade de divisões feitas nas imagens.
    """
    save_dataset(color, contrast, fold, format, height, images, minimum, model, name, n_features, output, patch, width)
    save_samples(images, input, output)


def save_dataset(color:str, contrast: str,
                 fold: int,
                 format: str,
                 height: int,
                 images: list,
                 minimum: int, model: str,
                 name: str,
                 n_features:int,
                 output: pathlib.Path | LiteralString | str, patch: int,
                 width: int):
    """
    Salva as informações do dataset.
    :param fold:  a classe que pertence aquelas imagens.
    :param features: quantidade de features extraídas.
    :param images: lista com as imagens que deverão ser salvas.
    :param patch: quantidade de divisões feitas nas imagens.
    :param output: local onde será salvo as imagens.
    """

    data = {
        'color': [color],
        'contrast': [contrast],
        'fold': [fold],
        'format': [format],
        'height': [height],
        'patch': [patch],
        'count_features': [n_features],
        'count_samples': [len(images)],
        'model': [model.name],
        'name': [name],
        'minimum': [minimum],
        'count_samples+patch': np.sum([len(image.patches) for image in images]),
        'width': [width],
    }
    df = pd.DataFrame(data, columns=list(data.keys()))
    filename = os.path.join(output, 'dataset.csv')
    df.to_csv(filename, sep=';', quoting=2, quotechar='"', encoding='utf-8', index=False, header=True)


def get_label(input: pathlib.Path | LiteralString | str, value: int) -> str:
    df = pd.read_csv(os.path.join(input, 'input.csv'), sep=';', header=0, index_col=None, engine='c', low_memory=False)
    df['output'] = df['output'].apply(lambda x: x.replace('f', ''))
    df['output'] = df['output'].astype('int64')
    return df[df['output'] == value]['input'].values[0]


def save_samples(images: list, input: pathlib.Path | LiteralString | str, output: pathlib.Path | LiteralString | str, ):
    """
    Salva as informações das amostras que foram extraídas as características.
    :param fold:  a classe que pertence aquelas imagens.
    :param features: quantidade de features extraídas.
    :param images: lista com as imagens que deverão ser salvas.
    :param patch: quantidade de divisões feitas nas imagens.
    :param output: local onde será salvo as imagens.
    """
    data = {'filename': [image.filename for image in sorted(images, key=lambda x: x.filename)],
            'fold': [image.fold for image in sorted(images, key=lambda x: x.filename)],
            'specific_epithet': [get_label(input, image.fold) for image in sorted(images, key=lambda x: x.filename)]}
    df = pd.DataFrame(data, columns=list(data.keys()))
    filename = os.path.join(output, 'samples.csv')
    df.to_csv(filename, sep=';', quoting=2, quotechar='"', encoding='utf-8', index=False, header=True)

def save(features: np.ndarray, fold: int, format: str,
         images: list,
         output: pathlib.Path | LiteralString | str, patch: int) -> None:
    """
    Chama as funções que salvam as features, as imagens e as informações da extração.
    :param fold: a classe que pertence aquelas imagens.
    :param features: matriz com as características extraídas.
    :param format: formato escolhido pelo usuário.
    :param images: lista com as imagens que deverão ser salvas.
    :param patch: quantidade de divisões feitas nas imagens.
    :param output:  local onde serão salvos as imagens e as features.
    """
    save_features(features, fold, format, patch, output)
    save_images(fold, images, output)

# def save(color:str, contrast:float, height:int, fold:int, minimum:int, model:str, name:str,
#          output: pathlib.Path | LiteralString | str,
#          patch:int, width:int):
#     save_csv(color, contrast, fold, height, minimum, model, name, output, patch, width)

# def save(descriptor: str, features: np.ndarray, images: list, output: pathlib.Path | LiteralString | str) -> None:
#     save_features(descriptor, features, output)
#     save_samples(images, output)
#     save_dataset(np.max([image.fold for image in images]), features, images, output, patch=1)
