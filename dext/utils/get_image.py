import numpy as np
from paz.processors.image import LoadImage


def get_image(raw_image_path):
    loader = LoadImage()
    raw_image = loader(raw_image_path)
    raw_image = raw_image.astype('uint8')
    if raw_image.shape[-1] != 3:
        raw_image = np.stack((raw_image,) * 3, axis=-1)
    return raw_image
