import cv2
import numpy as np
from paz.processors.image import LoadImage


def load_image_gray(filepath):
    """Load image from a ''filepath''.

    # Arguments
        filepath: String indicating full path to the image.
        num_channels: Int.

    # Returns
        Numpy array.
    """
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    image = np.expand_dims(image, -1)
    return image


def get_image(raw_image_path, load_type='rgb'):
    if load_type == 'rgb':
        loader = LoadImage()
        raw_image = loader(raw_image_path)
        raw_image = raw_image.astype('uint8')
        if raw_image.shape[-1] != 3:
            raw_image = np.stack((raw_image,) * 3, axis=-1)
        return raw_image
    else:
        return load_image_gray(raw_image_path)

