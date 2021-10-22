"""
Mask R-CNN
Common utility functions and classes.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import math
import numpy as np
import tensorflow as tf
import skimage.transform
import colorsys
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Polygon
from distutils.version import LooseVersion

import tensorflow.keras.backend as K
from tensorflow.keras.layers import ZeroPadding2D, MaxPooling2D
from tensorflow.keras.layers import Conv2D, Dense, Activation
from tensorflow.keras.layers import TimeDistributed, Lambda, Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization, Add
from tensorflow.keras.models import Model
from dext.model.faster_rcnn.layers import PyramidROIAlign


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.

    # Arguments:
        box: 1D vector [y_min, x_min, y_max, x_max]
        boxes: [boxes_count, (y_min, x_min, y_max, x_max)]
        box_area: float. the area of 'box'
        boxes_area: array of length boxes_count.

    # Returns:
        Intersection over union of given boxes
    """
    y_min = np.maximum(box[0], boxes[:, 0])
    y_max = np.minimum(box[2], boxes[:, 2])
    x_min = np.maximum(box[1], boxes[:, 1])
    x_max = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x_max - x_min, 0) * np.maximum(y_max - y_min, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes_A, boxes_B):
    """Computes IoU overlaps between two sets of boxes.

    # Arguments:
        boxes_A, boxes_B: [N, (y_min, x_min, y_max, x_max)].
    """
    area1 = (boxes_A[:, 2] - boxes_A[:, 0]) * (boxes_A[:, 3] - boxes_A[:, 1])
    area2 = (boxes_B[:, 2] - boxes_B[:, 0]) * (boxes_B[:, 3] - boxes_B[:, 1])

    overlaps = np.zeros((boxes_A.shape[0], boxes_B.shape[0]))
    for i in range(overlaps.shape[1]):
        box_B = boxes_B[i]
        overlaps[:, i] = compute_iou(box_B, boxes_A, area2[i], area1)
    return overlaps


def box_refinement(box, ground_truth_box):
    """Compute refinement needed to transform box to ground_truth_box.

    # Arguments:
        box: [N, (y_min, x_min, y_max, x_max)]
        ground_truth_box: [N, (y_min, x_min, y_max, x_max)]
                          (y_max, x_max) is assumed to be outside the box.
    """
    box = box.astype(np.float32)
    ground_truth_box = ground_truth_box.astype(np.float32)

    height = box[:, 2] - box[:, 0]
    width = box[:, 3] - box[:, 1]
    center_Y = box[:, 0] + 0.5 * height
    center_X = box[:, 1] + 0.5 * width

    ground_truth_H = ground_truth_box[:, 2] - ground_truth_box[:, 0]
    ground_truth_W = ground_truth_box[:, 3] - ground_truth_box[:, 1]
    groundtruth_center_Y = ground_truth_box[:, 0] + 0.5 * ground_truth_H
    groundtruth_center_X = ground_truth_box[:, 1] + 0.5 * ground_truth_W

    dY = (groundtruth_center_Y - center_Y) / height
    dX = (groundtruth_center_X - center_X) / width
    dH = np.log(ground_truth_H / height)
    dW = np.log(ground_truth_W / width)
    return np.stack([dY, dX, dH, dW], axis=1)


def resize_image(image, min_dim=None, max_dim=None,
                 min_scale=None, mode="square"):
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


def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """Generates anchor boxes

    # Arguments:
        scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
        ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
        shape: [height, width] spatial shape of the feature map over which
                to generate anchors.
        feature_stride: feature map stride relative to the image in pixels.
        anchor_stride: anchor stride on feature map. For example, if the
            value is 2 then generate anchors for every other feature map pixel.
    """
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)
    shifts_Y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_X = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_X, shifts_Y = np.meshgrid(shifts_X, shifts_Y)

    box_widths, box_center_X = np.meshgrid(widths, shifts_X)
    box_heights, box_center_Y = np.meshgrid(heights, shifts_Y)

    box_centers = np.stack(
        [box_center_Y, box_center_X], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    return boxes


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
       is associated with a level of the pyramid, but each ratio is used in
       all levels of the pyramid.

    # Returns:
        anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array.
                 Sorted with the same order of the given scales.
    """
    anchors = []
    for level in range(len(scales)):
        anchors.append(generate_anchors(scales[level], ratios,
                                        feature_shapes[level],
                                        feature_strides[level], anchor_stride))
    return np.concatenate(anchors, axis=0)


def get_resnet_features(input_image, architecture,
                        stage5=False, train_bn=True):
    """Builds ResNet graph.

    # Arguments:
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    x = ZeroPadding2D((3, 3))(input_image)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = Activation('relu')(x)
    C1 = x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1),
                   train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b',
                       train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c',
                            train_bn=train_bn)
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a',
                   train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b',
                       train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c',
                       train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d',
                            train_bn=train_bn)
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a',
                   train_bn=train_bn)
    block_count = {'resnet50': 5, 'resnet101': 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i),
                           train_bn=train_bn)
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a',
                       train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b',
                           train_bn=train_bn)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c',
                                train_bn=train_bn)
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]


def fpn_classifier_graph(
        rois, feature_maps, pool_size, num_classes, image_max_dim,
        train_bn=True, fc_layers_size=1024):
    """Builds the computation graph of the feature pyramid network classifier
       and regressor heads.

    # Arguments:
        rois: [batch, num_rois, (y_min, x_min, y_max, x_max)]
              Proposal boxes in normalized coordinates.
        feature_maps: List of feature maps from different pyramid layers,
                      [P2, P3, P4, P5].
        image_meta: [batch, (meta data)] Image details
        pool_size: The width of the square feature map generated from ROI Pool.
        num_classes: number of classes
        train_bn: Boolean. Train or freeze Batch Norm layers
        fc_layers_size: Size of the 2 FC layers

    # Returns:
        logits: classifier logits (before softmax)
                [batch, num_rois, NUM_CLASSES]
        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
                     Deltas to apply to proposal boxes
    """
    pool_size = pool_size
    num_classes = num_classes
    image_shape = (image_max_dim, image_max_dim, 3)
    image_shape = tf.convert_to_tensor(np.array(image_shape))
    x = PyramidROIAlign([pool_size, pool_size], name='roi_align_classifier')(
        [rois, image_shape] + feature_maps)
    conv_2d_layer = Conv2D(fc_layers_size, (pool_size, pool_size),
                           padding='valid')
    x = TimeDistributed(conv_2d_layer, name='mrcnn_class_conv1')(x)
    x = tf.reshape(x, [1000, x.shape[2], x.shape[3], x.shape[4]])
    x = tf.expand_dims(x, axis=0)
    x = TimeDistributed(BatchNorm(), name='mrcnn_class_bn1')(
        x, training=train_bn)
    x = Activation('relu')(x)
    x = TimeDistributed(Conv2D(fc_layers_size, (1, 1)),
                        name='mrcnn_class_conv2')(x)
    x = TimeDistributed(BatchNorm(), name='mrcnn_class_bn2')(
        x, training=train_bn)
    x = Activation('relu')(x)
    shared = Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),
                    name='pool_squeeze')(x)
    # Classifier head
    mrcnn_class_logits = TimeDistributed(Dense(num_classes),
                                         name='mrcnn_class_logits')(shared)
    mrcnn_probs = TimeDistributed(Activation('softmax'),
                                  name='mrcnn_class')(mrcnn_class_logits)
    # Bounding box head
    x = TimeDistributed(Dense(num_classes * 4, activation='linear'),
                        name='mrcnn_bbox_fc')(shared)
    s = K.int_shape(x)
    mrcnn_bbox = Reshape((s[1], num_classes, 4), name='mrcnn_bbox')(x)
    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_rpn_model(anchor_stride, anchors_per_location, depth):
    """Builds a Keras model of the Region Proposal Network.

    # Arguments:
        anchors_per_location: number of anchors per pixel in feature map
        anchor_stride: Anchor for every pixel in feature map
                       Typically 1 or 2
        depth: Depth of the backbone feature map.

    # Returns:
        rpn_class_logits: [batch, H * W * anchors_per_location, 2]
                          Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * anchors_per_location, 2]
                   Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * anchors_per_location,
                  (dy, dx, log(dh), log(dw))] Deltas to be applied to anchors.
    """
    input_feature_map = Input(shape=[None, None, depth],
                              name='input_rpn_feature_map')
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return Model([input_feature_map], outputs, name='rpn_model')


def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """Builds the computation graph of Region Proposal Network.

    # Arguments:
        feature_map: backbone features [batch, height, width, depth]
        anchors_per_location: number of anchors per pixel in feature map
        anchor_stride: Typically 1 (anchors for every pixel in feature map),
                       or 2 (every other pixel).

    # Returns:
        rpn_class_logits: [batch, H * W * anchors_per_location, 2]
                           Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * anchors_per_location, 2]
                    Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * anchors, (dy, dx, log(dh), log(dw))]
                   Deltas to be applied to anchors.
    """
    shared = Conv2D(512, (3, 3), padding='same', activation='relu',
                    strides=anchor_stride,
                    name='rpn_conv_shared')(feature_map)
    x = Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
               activation='linear', name='rpn_class_raw')(shared)
    rpn_class_logits = Lambda(lambda t: tf.reshape(t,
                              [tf.shape(t)[0], -1, 2]))(x)
    rpn_probs = Activation('softmax', name='rpn_class_xxx')(rpn_class_logits)
    x = Conv2D(anchors_per_location * 4, (1, 1), padding='valid',
               activation='linear', name='rpn_bbox_pred')(shared)
    rpn_bbox = Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)
    return [rpn_class_logits, rpn_probs, rpn_bbox]


def trim_zeros(x):
    """Removes rows that are all zeros.

    # Arguments:
        x: [rows, columns]
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]


def norm_boxes(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [N, (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)


def denorm_boxes(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [N, (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)


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


class BatchNorm(BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.
    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """
    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making
            inferences
        """
        return super(self.__class__, self).call(inputs, training=training)


def compute_backbone_shapes(image_shape, backbone, compute_backbone_shape,
                            backbone_strides):
    """Computes the width and height of each stage of the backbone network.
    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    if callable(backbone):
        return compute_backbone_shape(image_shape)

    # Currently supports ResNet only
    assert backbone in ["resnet50", "resnet101"]
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            for stride in backbone_strides])


def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(),
                                                           array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("", ""))
        text += "  {}".format(array.dtype)
    print(text)


def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array.
    image_id: An int ID of the image. Useful for debugging.
    original_image_shape: [H, W, C] before resizing or padding.
    image_shape: [H, W, C] after resizing and padding
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    scale: The scaling factor applied to the original image (float32)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +                  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +           # size=3
        list(window) +        # size=4 (y1, x1, y2, x2) in image cooredinates
        [scale] +                     # size=1
        list(active_class_ids)        # size=num_classes
    )
    return meta


def parse_image_meta(meta):
    """Parses an array that contains image attributes to its components.
    See compose_image_meta() for more details.
    meta: [batch, meta length] where meta length depends on NUM_CLASSES
    Returns a dict of the parsed values.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id.astype(np.int32),
        "original_image_shape": original_image_shape.astype(np.int32),
        "image_shape": image_shape.astype(np.int32),
        "window": window.astype(np.int32),
        "scale": scale.astype(np.float32),
        "active_class_ids": active_class_ids.astype(np.int32),
    }


def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.
    meta: [batch, meta length] where meta length depends on NUM_CLASSES
    Returns a dict of the parsed tensors.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }


def normalize_image(images, mean_pixel):
    """Expects an RGB image (or array of images) and subtracts
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - mean_pixel


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)


def norm_boxes_graph(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels
    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.
    Returns:
        [..., (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)


def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main
        path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
               use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
               name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
               use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    x = Add()([x, input_tensor])
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main
        path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with
    subsample=(2,2) And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), strides=strides,
               name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
               name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base +
               '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                      name=conv_name_base + '1',
                      use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = Add()([x, shortcut])
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


############################################################
#  Visualization
############################################################

def display_images(images, titles=None, cols=7, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interpolation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    np.random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = len(boxes)
    if not N:
        print("\n*** No instances to display *** \n")
    # else:
    #     assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        fig, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")
        if masks:
            # Mask
            mask = masks[:, :, i]
            if show_mask:
                masked_image = apply_mask(masked_image, mask, color)

            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)

    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()
        plt.savefig('mask.jpg')
