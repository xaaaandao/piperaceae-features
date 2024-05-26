import dataclasses
import os
import pathlib

import PIL
import tensorflow as tf

from PIL import ImageEnhance
from typing import LiteralString


def adjust_contrast(contrast: float, image: PIL.Image) -> ImageEnhance.Contrast:
    """
    Altera o valor do contraste da imagem.
    :param contrast: valor para ajustar o contraste.
    :param image: imagem que tera seu contraste ajustado.
    :return: imagem com o constrate ajustado.
    """
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    return image


@dataclasses.dataclass
class Image:
    filename: str = dataclasses.field(init=False)
    extension: str = dataclasses.field(init=False)
    path: str
    patches: list = dataclasses.field(default_factory=list)

    def __post_init__(self):
        """
        Transforma o caminho da imagem em pathlib.Path. Essa transformação permite extrair informações, como o nome do arquivo e a sua extensão.
        """
        p = pathlib.Path(self.path)
        self.filename = p.stem
        self.extension = p.suffix

    def print(self):
        print(self.filename, self.extension, self.path, sep='\n')

    def save_patches(self, output: pathlib.Path | LiteralString | str) -> None:
        """
        Salva as divisões (ou patches) das imagens.
        :param output: local onde serão salvos os patches.
        :return: .
        """
        if len(self.patches) <= 0:
            raise ValueError('No patches to save')

        p = os.path.join(output, self.filename)
        os.makedirs(p, exist_ok=True)

        for i, patch in enumerate(self.patches, start=1):
            output_filename = self.filename + '-' + str(i) + self.extension
            output_filename = os.path.join(p, output_filename)
            tf.keras.preprocessing.image.save_img(output_filename, patch)