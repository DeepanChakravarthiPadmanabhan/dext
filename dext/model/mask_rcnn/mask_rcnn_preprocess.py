from paz.abstract import Processor
from dext.model.mask_rcnn.utils import resize_image, normalize_image


class NormalizeImages(Processor):
    def __init__(self, config):
        self.config = config
        super(NormalizeImages, self).__init__()

    def call(self, images, windows):
        normalized_images = []
        for image in images:
            molded_image = normalize_image(image, self.config)
            normalized_images.append(molded_image)
        return normalized_images, windows


class ResizeImages(Processor):
    def __init__(self, config):
        self.IMAGE_MIN_DIM = config.IMAGE_MIN_DIM
        self.IMAGE_MIN_SCALE = config.IMAGE_MIN_SCALE
        self.IMAGE_MAX_DIM = config.IMAGE_MAX_DIM
        self.IMAGE_RESIZE_MODE = config.IMAGE_RESIZE_MODE

    def call(self, images):
        resized_images, windows = [], []
        for image in images:
            resized_image, window, _, _, _ = resize_image(
                image,
                min_dim=self.IMAGE_MIN_DIM,
                min_scale=self.IMAGE_MIN_SCALE,
                max_dim=self.IMAGE_MAX_DIM,
                mode=self.IMAGE_RESIZE_MODE)
            resized_images.append(resized_image)
            windows.append(window)
        return resized_images, windows
