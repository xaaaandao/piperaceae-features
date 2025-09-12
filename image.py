import dataclasses
import os
import pathlib
import tensorflow as tf

from typing import LiteralString

@dataclasses.dataclass
class Image:
    filename : pathlib.Path
    fold : int
    patches: list

    def __post_init__(self):
        """
        Transforma o caminho da imagem em pathlib.Path. Essa transformação permite extrair informações, como o nome do arquivo e a sua extensão.
        """
        p = pathlib.Path(self.filename)
        self.filename = p.stem
        self.extension = p.suffix
        self.fold = self.set_fold(p.parent.name)

    def set_fold(self, p: pathlib.Path) -> int:
        return int(float(str(p).replace('f', '').lstrip('0')))

    def print(self):
        print(self.filename, self.extension, self.path, sep="\n")

    def save(self, output: pathlib.Path | LiteralString | str) -> None:
        """
        Salva as divisões (ou patches) das imagens.
        :param output: local onde serão salvos os patches.
        :return: .
        """
        if len(self.patches) <= 0:
            raise ValueError("No patches to save")

        p = os.path.join(output, self.filename)
        os.makedirs(p, exist_ok=True)

        for i, patch in enumerate(self.patches, start=1):
            output_filename = self.filename + '-' + str(i) + self.extension
            output_filename = os.path.join(p, output_filename)
            if not os.path.exists(output_filename):
                tf.keras.preprocessing.image.save_img(output_filename, patch)