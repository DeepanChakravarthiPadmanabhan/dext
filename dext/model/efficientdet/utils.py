import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Reshape, Concatenate, Flatten, Activation

import paz.processors as pr
from paz.abstract import SequentialProcessor
from paz.processors.image import RGB_IMAGENET_MEAN, RGB_IMAGENET_STDEV


def get_activation(features, activation):
    """Apply non-linear activation function to features provided.

    # Arguments
        features: Tensor, representing an input feature map
        to be pass through an activation function.
        activation: A string specifying the activation function
        type.

    # Returns
        activation function: features transformed by the
        activation function.
    """
    if activation in ('silu', 'swish'):
        return tf.nn.swish(features)
    elif activation == 'relu':
        return tf.nn.relu(features)
    else:
        raise ValueError('Unsupported activation fn {}'.format(activation))


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


def efficientdet_preprocess(image, image_size):
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
    if type(image_size) == tuple:
        image_size = image_size[0]
    preprocessing = SequentialProcessor([
        pr.CastImage(float),
        pr.SubtractMeanImage(mean=RGB_IMAGENET_MEAN),
        pr.DivideStandardDeviationImage(standard_deviation=RGB_IMAGENET_STDEV),
        pr.ScaledResize(image_size=image_size),
        ])
    image, image_scale = preprocessing(image)
    return image, image_scale


def create_multibox_head(class_outputs, box_outputs, model, num_regressions=4):
    num_levels = model.num_levels
    num_classes = model.num_classes
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
