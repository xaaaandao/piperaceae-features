import os
import pathlib
from typing import LiteralString

import numpy as np
import pandas as pd


def save_features(features, fold: int, model: str, output: pathlib.Path | LiteralString | str, patch: int) -> None:
    """
    Cria uma pasta para salvar as features no formato npy de um determinado fold.
    :param features: matriz com as características extraídas.
    :param fold: a classe que pertence aquelas features.
    :param patch: quantidade de divisões feitas nas imagens.
    :param output: local onde será salvo as features.
    :return: .
    """
    output = os.path.join(output, "features", model, "f%d" % fold)
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


def get_label(input: pathlib.Path | LiteralString | str, images: list) -> list:
    filename = os.path.join(input, "input.csv")
    if os.path.exists(filename):
        labels = []
        df = pd.read_csv(filename, sep=';', header=0, index_col=None, engine='c', low_memory=False)
        df["output"] = df["output"].apply(lambda x: x.replace('f', ''))
        df["output"] = df["output"].astype("int64")
        for image in sorted(images, key=lambda x: x.filename):
            labels.append(df[df["output"] == image.fold]["input"].values[0])
        return labels
    return [None] * len(images)


def save_samples(images: list, input: pathlib.Path | LiteralString | str, model: str,
                 output: pathlib.Path | LiteralString | str):
    """
    Salva as informações das amostras que foram extraídas as características.
    :param fold:  a classe que pertence aquelas imagens.
    :param features: quantidade de features extraídas.
    :param images: lista com as imagens que deverão ser salvas.
    :param patch: quantidade de divisões feitas nas imagens.
    :param output: local onde será salvo as imagens.
    """
    data = {"filename": [image.filename for image in sorted(images, key=lambda x: x.filename)],
            "fold": [image.fold for image in sorted(images, key=lambda x: x.filename)],
            "specific_epithet": get_label(input, images)}
    df = pd.DataFrame(data, columns=list(data.keys()))
    filename = os.path.join(output, "features", model, "samples.csv")
    df.to_csv(filename, sep=';', quoting=2, quotechar='"', encoding="utf-8", index=False, header=True)
