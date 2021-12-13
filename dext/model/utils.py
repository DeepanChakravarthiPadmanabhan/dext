import cv2
import numpy as np
from paz.abstract.processor import Processor


def get_layers_and_length(layer):
    if hasattr(layer, 'layers'):
        block_length = len(layer.layers)
        block_layers = layer.layers
    else:
        block_length = 0
        block_layers = layer
    return block_length, block_layers


def get_all_layers(model):
    all_layers = []
    for i in model.layers[1:]:
        block_length, block_layers = get_layers_and_length(i)
        if block_length:
            all_layers.extend(block_layers)
        else:
            all_layers.append(block_layers)
    return all_layers


def find_image_scale(input_image_shape, processed_image_shape):
    input_h, input_w, _ = input_image_shape
    _, processed_h, processed_w, _ = processed_image_shape
    image_scale_y = np.array(processed_h).astype('float32') / input_h
    image_scale_x = np.array(processed_w).astype('float32') / input_w
    return 1/image_scale_y, 1/image_scale_x


def resize_image(image, size):
    """Resize image.

    # Arguments
        image: Numpy array.
        dtype: List of two ints.

    # Returns
        Numpy array.
    """
    if(type(image) != np.ndarray):
        raise ValueError(
            'Recieved Image is not of type numpy array', type(image))
    else:
        if (image.shape[0] != size[0]) or (image.shape[1] != size[1]):
            new_image = cv2.resize(image, size)
        else:
            new_image = image
        return new_image


class ResizeImage(Processor):
    """Resize image.

    # Arguments
        size: List of two ints.
    """
    def __init__(self, shape):
        self.shape = shape
        super(ResizeImage, self).__init__()

    def call(self, image):
        return resize_image(image, self.shape)
