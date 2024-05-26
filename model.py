import math
from typing import Tuple

import tensorflow as tf


def get_model(model, **kwargs):
    """
    Retorna a rede neural convolucional (CNN) com os parâmetros que foram enviados.
    :param model: modelo que foi solicitado pelo usuário.
    :param kwargs: argumentos que foram enviados pelo usuário.
    :return: CNN escolhida pelo usuário.
    """
    match model:
        case 'vgg16':
            return tf.keras.applications.vgg16.VGG16(**kwargs), tf.keras.applications.vgg16.preprocess_input
        case 'resnet50v2':
            return tf.keras.applications.resnet_v2.ResNet50V2(
                **kwargs), tf.keras.applications.resnet_v2.preprocess_input
        case 'mobilenetv2':
            return tf.keras.applications.mobilenet_v2.MobileNetV2(
                **kwargs), tf.keras.applications.mobilenet_v2.preprocess_input

    raise ValueError


def get_input_shape(orientation: str, patch: int, spec_height: int, spec_width: int)-> Tuple:
    """
    Calcula o corte da imagem baseado na orientação.
    :param orientation: orientação do corte da imagem.
    :param patch: quantidade de divisões na imagem.
    :param spec_height: altura da imagem.
    :param spec_width: largura da imagem.
    :return: tupla com os valores de altura e largura da imagem.
    """
    match orientation:
        case 'horizontal':
            return math.floor(spec_height / patch), spec_width, 3
        case 'vertical':
            return spec_height, math.floor(spec_width / patch), 3
