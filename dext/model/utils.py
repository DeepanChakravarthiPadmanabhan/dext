import cv2
import numpy as np
from paz.abstract.processor import Processor
import skimage.transform
from distutils.version import LooseVersion


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
    if (type(image) != np.ndarray):
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


def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False,
           anti_aliasing_sigma=None):
    """A wrapper for Scikit-Image resize().

    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)

def resize_image_fasterrcnn(image, min_dim=512, max_dim=512, min_scale=0, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.

    # Arguments:
        min_dim: Minimum dimension of image to resize
        max_dim: Maximum dimension of image to resize
        min_scale: Image scale percentage
        mode: Resizing mode. e.g. None, square, pad64 or crop

    # Returns:
        image: Resized image
        window: Coordinates of unpadded image (y_min, x_min, y_max, x_max)
        padding: Padding added to the image
            [(top, bottom), (left, right), (0, 0)]
    """
    image_dtype = image.dtype
    H, W = image.shape[:2]
    window = (0, 0, H, W)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == 'none':
        return image, window, scale, padding, crop

    if min_dim:
        scale = max(1, min_dim / min(H, W))
    if min_scale and scale < min_scale:
        scale = min_scale

    if max_dim and mode == 'square':
        image_max = max(H, W)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    if scale != 1:
        image = resize(image, (round(H * scale), round(W * scale)),
                       preserve_range=True)
    H, W = image.shape[:2]
    top_pad = (max_dim - H) // 2
    bottom_pad = max_dim - H - top_pad
    left_pad = (max_dim - W) // 2
    right_pad = max_dim - W - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(image, padding, mode='constant', constant_values=0)
    window = (top_pad, left_pad, H + top_pad, W + left_pad)
    return image.astype(image_dtype), window, scale, padding, crop


def resize_image_general(raw_image, image_size, model_name):
    if model_name == 'FasterRCNN':
        image,  window, _, _, _ = resize_image_fasterrcnn(
            raw_image, image_size[0], image_size[1])
    else:
        image = cv2.resize(raw_image, image_size)
    return image
