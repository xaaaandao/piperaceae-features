import dataclasses
import os
import pathlib
from typing import LiteralString

import pandas as pd

from model import Model, get_input_shape


@dataclasses.dataclass(init=False)
class Dataset:
    fold: int
    height: int
    images: list
    input_dir: str
    model: Model
    name: str
    n_features: int
    n_samples: int
    n_samples_patch: int
    orientation: str
    patch: int
    width: int

    def __init__(self, color, gpuid, height, input_dir, model, orientation, patch, width, name=None):
        self.color = color
        self.gpuid = gpuid
        self.height = height
        self.images = list()
        self.input_dir = input_dir
        self.model = Model(model, self.gpuid)
        self.name = name
        self.n_samples = 0
        self.n_samples_patch = 0
        self.orientation = orientation
        self.patch = patch
        self.width = width

        input_shape = get_input_shape(self.orientation, self.patch, self.height, self.width)
        self.model.set_model(weights="imagenet", include_top=False, input_shape=input_shape, pooling="avg")

    def save(self, output):
        data = {
            "height": [self.height],
            "patch": [self.patch],
            "n_features": [self.n_features],
            "n_labels": [self.fold],
            "n_samples": [self.n_samples_patch / self.patch],
            "n_samples+patch": [self.n_samples_patch],
            "model": [self.model.name],
            "name": [self.name],
            "width": [self.width],
        }
        df = pd.DataFrame(data, columns=list(data.keys()))
        df = df.transpose()
        filename = os.path.join(output, "features", self.color, str(self.width), self.model.name, "dataset.csv")
        print("saving %s" % filename)
        df.to_csv(filename, sep=';', quoting=2, quotechar='"', encoding="utf-8", index=True, header=False)

    def update(self, features_shape: tuple, fold: int, patch: int) -> None:
        self.fold = fold
        # -2, porque eu adicionei duas colunas (classe e nome do arquivo)
        self.n_features = features_shape[1] - 2
        self.n_samples_patch = self.n_samples_patch + features_shape[0]
        self.patch = patch

    def save_patches(self, output):
        for image in self.images:
            p = os.path.join(output, "images", self.color, str(self.width), image.specific_epithet)# "f%d" % image.fold)
            os.makedirs(p, exist_ok=True)
            image.save(p)

    def save_samples(self, input, output):
        data = {"filename": [image.filename for image in sorted(self.images, key=lambda x: x.filename)],
                "fold": [image.fold for image in sorted(self.images, key=lambda x: x.filename)],
                "specific_epithet": [image.specific_epithet for image in sorted(self.images, key=lambda x: x.filename)]}
        df = pd.DataFrame(data, columns=list(data.keys()))
        filename = os.path.join(output, "samples.csv")
        print("saving %s" % filename)
        df.to_csv(filename, sep=';', quoting=2, quotechar='"', encoding="utf-8", index=False, header=True)


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
