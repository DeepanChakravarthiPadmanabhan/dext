import tensorflow as tf
from dext.interpretation_method.convert_box.ssd_conversion import (decode,
                                                                   scale_box)


def process_outputs(outputs):
    boxes, classes = outputs[:, :4], outputs[:, 4:]
    s1, s2, s3, s4 = tf.split(boxes, num_or_size_splits=4, axis=1)
    boxes = tf.concat([s2, s1, s4, s3], axis=1)
    boxes = tf.expand_dims(boxes, axis=0)
    classes = tf.expand_dims(classes, axis=0)
    outputs = tf.concat([boxes, classes], axis=2)
    return outputs


def get_efficientdet_boxes(conv_outs, prior_boxes, image_size, image_scale,
                           to_ic):
    conv_outs = process_outputs(conv_outs)
    conv_outs = conv_outs[0]
    conv_outs = decode(conv_outs, prior_boxes, variances=[1, 1, 1, 1])
    image_size = tf.constant(image_size, dtype=tf.float32)
    conv_outs = tf.math.divide(conv_outs, image_size)
    conv_outs = conv_outs[:, :4]
    if to_ic:
        # Anchor boxes are in the image coordinates level
        conv_outs = tf.math.multiply(conv_outs, image_size)
        conv_outs = scale_box(conv_outs, image_scale)
    return conv_outs


def efficientdet_convert_coordinates(conv_outs, prior_boxes, visualize_index,
                                     image_size, image_scale, to_ic=False):
    conv_outs = get_efficientdet_boxes(conv_outs, prior_boxes, image_size,
                                       image_scale, to_ic)
    conv_outs = conv_outs[visualize_index[1]]
    return conv_outs
