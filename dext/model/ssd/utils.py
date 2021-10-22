import numpy as np
from paz import processors as pr
from paz.abstract import SequentialProcessor
from paz.backend.image import resize_image
from dext.model.utils import ResizeImage


def find_image_scale(input_image_shape, processed_image_shape):
    input_h, input_w, _ = input_image_shape
    _, processed_h, processed_w, _ = processed_image_shape
    image_scale_y = np.array(processed_h).astype('float32') / input_h
    image_scale_x = np.array(processed_w).astype('float32') / input_w
    return 1/image_scale_y, 1/image_scale_x


def ssd_preprocess(image, image_size, only_resize=False):
    input_image_shape = image.shape
    if type(image_size) == int:
        image_size = (image_size, image_size)
    if only_resize:
        preprocessing = SequentialProcessor([ResizeImage(image_size),
                                             pr.CastImage(float),
                                             pr.ExpandDims(axis=0)])
    else:
        preprocessing = SequentialProcessor([
            ResizeImage(image_size),
            pr.SubtractMeanImage(mean=pr.RGB_IMAGENET_MEAN),
            pr.CastImage(float), pr.ExpandDims(axis=0)])
    image = preprocessing(image)
    processed_image_shape = image.shape
    image_scale = find_image_scale(input_image_shape, processed_image_shape)
    return image, image_scale


def scaled_resize(image, image_size):
    """
    # Arguments
        image: Numpy array, raw input image.
    """
    crop_offset_y = np.array(0)
    crop_offset_x = np.array(0)
    height = np.array(image.shape[0]).astype('float32')
    width = np.array(image.shape[1]).astype('float32')
    image_scale_y = np.array(image_size).astype('float32') / height
    image_scale_x = np.array(image_size).astype('float32') / width
    image_scale = np.minimum(image_scale_x, image_scale_y)
    scaled_height = (height * image_scale).astype('int32')
    scaled_width = (width * image_scale).astype('int32')
    scaled_image = resize_image(image, (scaled_width, scaled_height))
    scaled_image = scaled_image[
                   crop_offset_y: crop_offset_y + image_size,
                   crop_offset_x: crop_offset_x + image_size,
                   :]
    output_images = np.zeros((image_size,
                              image_size,
                              image.shape[2]))
    s_h = scaled_image.shape[0]
    s_w = scaled_image.shape[1]
    s_c = scaled_image.shape[2]
    output_images[:s_h, :s_w, :s_c] = scaled_image
    image_scale = 1 / image_scale
    return output_images, image_scale
