import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Reshape, Concatenate, Flatten, Activation

import paz.processors as pr
from paz.abstract import SequentialProcessor
from dext.model.utils import ResizeImage


def get_drop_connect(features, is_training, survival_rate):
    """Drop the entire conv with given survival probability.
    Deep Networks with Stochastic Depth, https://arxiv.org/pdf/1603.09382.pdf

    # Arguments
        features: Tensor, input feature map to undergo
        drop connection.
        is_training: Bool specifying the training phase.
        survival_rate: Float, survival probability to drop
        input convolution features.

    # Returns
        output: Tensor, output feature map after drop connect.
    """
    if not is_training:
        return features
    batch_size = tf.shape(features)[0]
    random_tensor = survival_rate
    random_tensor += tf.random.uniform([batch_size, 1, 1, 1],
                                       dtype=features.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = features / survival_rate * binary_tensor
    return output


def find_image_scale(input_image_shape, processed_image_shape):
    input_h, input_w, _ = input_image_shape
    _, processed_h, processed_w, _ = processed_image_shape
    image_scale_y = np.array(processed_h).astype('float32') / input_h
    image_scale_x = np.array(processed_w).astype('float32') / input_w
    return 1/image_scale_y, 1/image_scale_x


def efficientdet_preprocess(image, image_size, only_resize=False):
    """
    Preprocess image for EfficientDet model.

    # Arguments
        image: Tensor, raw input image to be preprocessed
        of shape [bs, h, w, c]
        image_size: Tensor, size to resize the raw image
        of shape [bs, new_h, new_w, c]

    # Returns
        image: Numpy array, resized and preprocessed image
        image_scale: Numpy array, scale to reconstruct each of
        the raw images to original size from the resized
        image.
    """
    input_image_shape = image.shape
    if type(image_size) == int:
        image_size = (image_size, image_size)
    if only_resize:
        preprocessing = SequentialProcessor([
            ResizeImage(image_size),
            pr.CastImage(float),
            pr.ExpandDims(axis=0)])
    else:
        preprocessing = SequentialProcessor([
            ResizeImage(image_size),
            pr.SubtractMeanImage(mean=pr.RGB_IMAGENET_MEAN),
            pr.DivideStandardDeviationImage(
                standard_deviation=pr.RGB_IMAGENET_STDEV),
            pr.CastImage(float),
            pr.ExpandDims(axis=0)
            ])
    image = preprocessing(image)
    processed_image_shape = image.shape
    image_scale = find_image_scale(input_image_shape, processed_image_shape)
    return image, image_scale


def create_multibox_head(branch_tensors, num_levels, num_classes,
                         num_regressions=4):
    class_outputs = branch_tensors[0]
    box_outputs = branch_tensors[1]
    classification_layers, regression_layers = [], []
    for level in range(0, num_levels):
        class_leaf = class_outputs[level]
        class_leaf = Flatten()(class_leaf)
        classification_layers.append(class_leaf)

        regress_leaf = box_outputs[level]
        regress_leaf = Flatten()(regress_leaf)
        regression_layers.append(regress_leaf)

    classifications = Concatenate(axis=1)(classification_layers)
    regressions = Concatenate(axis=1)(regression_layers)
    num_boxes = K.int_shape(regressions)[-1] // num_regressions
    classifications = Reshape((num_boxes, num_classes))(classifications)
    classifications = Activation('sigmoid')(classifications)
    regressions = Reshape((num_boxes, num_regressions))(regressions)
    outputs = Concatenate(axis=2, name='boxes')([regressions, classifications])
    return outputs
