import os
import pathlib
from typing import LiteralString

import numpy as np
import pandas as pd


def save_features(dataset, features, fold: int, model: str, output: pathlib.Path | LiteralString | str, patch: int) -> None:
    """
    Cria uma pasta para salvar as features no formato npy de um determinado fold.
    :param features: matriz com as características extraídas.
    :param fold: a classe que pertence aquelas features.
    :param patch: quantidade de divisões feitas nas imagens.
    :param output: local onde será salvo as features.
    :return: .
    """
    output = os.path.join(output, "features", dataset.color, str(dataset.width), model, "f%d" % fold)
    os.makedirs(output, exist_ok=True)
    filename = f"fold-%d_patches-%d.npy" % (fold, patch)
    np.save(os.path.join(output, filename), features, allow_pickle=True)
    print("saving %s" % os.path.join(output, filename))


def save_patches(images: list, output: pathlib.Path | LiteralString | str) -> None:
    """
    Cria uma pasta para salvar as imagens que foram divididas.
    :param fold: a classe que pertence aquelas imagens.
    :param images: lista com as imagens que deverão ser salvas.
    :param output: local onde será salvo as imagens.
    """
    if len(images) <= 0:
        raise ValueError("No images found")

    for image in images:
        p = os.path.join(output, "images", "f%d" % image.fold)
        os.makedirs(p, exist_ok=True)
        image.save(p)


