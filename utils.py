import numpy as np

STAGE = [-2000, -800, 150, 6000]
CLIP = [-np.inf] + STAGE[1:-1] + [np.inf]


def normalize(image):
    norms = []
    for i in range(len(CLIP) - 1):
        norm = np.clip(image, *CLIP[i:i + 2])
        norms.append(2 * (norm - STAGE[i]) / (STAGE[i + 1] - STAGE[i]) - 1)
    return np.stack(norms, axis=0)


def denormalize(image):
    result = 0 - sum(STAGE[1:-1])
    for i in range(len(CLIP) - 1):
        result += (image[i] + 1) / 2 * (STAGE[i + 1] - STAGE[i]) + STAGE[i]
    return result


def load(file, size):
    if isinstance(size, int):
        size = size, size
    return np.fromfile(file, dtype='float32').reshape(*size)
