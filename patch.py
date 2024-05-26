import math


def next_patch_horizontal(spec: int, n: int):
    """
    Retorna a posição do próximo corte na imagem.
    :param spec: é a altura da imagem.
    :param n: é o número de divisões que serão feitas na imagem.
    """
    step = math.floor(spec.shape[0] / n)
    for i in range(n):
        yield spec[i * step:(i + 1) * step, :, :]


def next_patch_vertical(spec: int, n: int):
    """
    Retorna a posição do próximo corte na imagem.
    :param spec: é a altura da imagem.
    :param n: é o número de divisões que serão feitas na imagem.
    """
    step = math.floor(spec.shape[1] / n)
    for i in range(n):
        yield spec[:, i * step:(i + 1) * step, :]


def next_patch(spec: int, n: int, orientation: str):
    """
    Chama a função baseado na orientação em que serão feito a divisão da imagem.
    :param spec: é a altura da imagem.
    :param n: é o número de divisões que serão feitas na imagem.
    :param orientation: é a orientação do corte da imagem.
    """
    match orientation:
        case 'horizontal':
            return next_patch_horizontal(spec, n)
        case 'vertical':
            return next_patch_vertical(spec, n)
