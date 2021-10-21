import numpy as np
from paz.abstract import Processor
from paz.abstract import SequentialProcessor

from dext.model.faster_rcnn.utils import resize_image, normalize_image


class NormalizeImages(Processor):
    def __init__(self, mean_pixel):
        self.mean_pixel = mean_pixel
        super(NormalizeImages, self).__init__()

    def call(self, image, window):
        normalized_image = normalize_image(image, self.mean_pixel)
        return normalized_image, window


class ResizeImages(Processor):
    def __init__(self, image_min_dim, image_min_scale, image_max_dim,
                 image_resize_mode):
        self.IMAGE_MIN_DIM = image_min_dim
        self.IMAGE_MIN_SCALE = image_min_scale
        self.IMAGE_MAX_DIM = image_max_dim
        self.IMAGE_RESIZE_MODE = image_resize_mode

    def call(self, image):
        resized_image, window, _, _, _ = resize_image(
            image,
            min_dim=self.IMAGE_MIN_DIM,
            min_scale=self.IMAGE_MIN_SCALE,
            max_dim=self.IMAGE_MAX_DIM,
            mode=self.IMAGE_RESIZE_MODE)
        return resized_image, window


def faster_rcnn_preprocess(image, image_size):
    image_min_dim = 800
    image_min_scale = 0
    image_max_dim = 1024
    image_resize_mode = 'square'
    mean_pixel = np.array([123.7, 116.8, 103.9])
    if image_size < image_max_dim:
        image_max_dim = image_size
        image_min_dim = image_size
    preprocess = SequentialProcessor([
        ResizeImages(image_min_dim, image_min_scale, image_max_dim,
                     image_resize_mode),
        NormalizeImages(mean_pixel)])
    normalized_image, window = preprocess(image)
    normalized_image = normalized_image[np.newaxis]
    return normalized_image, window
