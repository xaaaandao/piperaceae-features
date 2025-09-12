import dataclasses
import os

import pandas as pd


@dataclasses.dataclass(init=False)
class Dataset:
    fold: int
    height: int
    input_dir: str
    model: str
    name: str
    n_features: int
    n_samples: int
    n_samples_patch: int
    orientation: str
    patch: int
    width: int

    def __init__(self, height, input_dir, model, orientation, width, name=None):
        self.height = height
        self.input_dir = input_dir
        self.model = model
        self.name = name
        self.orientation = orientation
        self.width = width

    def save(self, output):
        data = {
             "fold": [self.fold],
             "height": [self.height],
             "patch": [self.patch],
             "n_features": [self.n_features],
             "n_samples": [self.n_samples],
             "n_samples+patch": [self.n_samples_patch],
             "model": [self.model],
             "name": [self.name],
             "width": [self.width],
        }
        df = pd.DataFrame(data, columns=list(data.keys()))
        filename = os.path.join(output, "features", self.model,  "dataset.csv")
        df.to_csv(filename, sep=';', quoting=2, quotechar='"', encoding="utf-8", index=True, header=False)

    def update(self, features_shape: tuple, fold: int, patch: int) -> None:
        self.fold = fold
        self.n_features = features_shape[1]
        self.n_samples = features_shape[0] / patch
        self.n_samples_patch = features_shape[0]
        self.patch = patch

