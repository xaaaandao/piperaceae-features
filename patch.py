import math


def next_patch_horizontal(spec, n):
    step = math.floor(spec.shape[0] / n)
    for i in range(n):
        yield spec[i * step:(i + 1) * step, :, :]


def next_patch_vertical(spec, n):
    step = math.floor(spec.shape[1] / n)
    for i in range(n):
        yield spec[:, i * step:(i + 1) * step, :]


def next_patch(spec, n, orientation):
    match orientation:
        case 'horizontal':
            return next_patch_horizontal(spec, n)
        case 'vertical':
            return next_patch_vertical(spec, n)