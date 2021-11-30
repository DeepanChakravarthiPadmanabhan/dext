import tensorflow as tf


def to_corner_form(boxes):
    """Transform from center coordinates to corner coordinates.

    # Arguments
        boxes: Numpy array with shape `(num_boxes, 4)`.

    # Returns
        Numpy array with shape `(num_boxes, 4)`.
    """
    center_x, center_y = boxes[:, 0:1], boxes[:, 1:2]
    W, H = boxes[:, 2:3], boxes[:, 3:4]
    x_min = center_x - (W / 2.0)
    x_max = center_x + (W / 2.0)
    y_min = center_y - (H / 2.0)
    y_max = center_y + (H / 2.0)
    return tf.concat([x_min, y_min, x_max, y_max], axis=1)


def decode(predictions, priors, variances=[0.1, 0.1, 0.2, 0.2]):
    """Decode default boxes into the ground truth boxes

    # Arguments
        loc: Numpy array of shape `(num_priors, 4)`.
        priors: Numpy array of shape `(num_priors, 4)`.
        variances: List of two floats. Variances of prior boxes.

    # Returns
        decoded boxes: Numpy array of shape `(num_priors, 4)`.
    """
    center_x = predictions[:, 0:1] * priors[:, 2:3] * variances[0]
    center_x = center_x + priors[:, 0:1]
    center_y = predictions[:, 1:2] * priors[:, 3:4] * variances[1]
    center_y = center_y + priors[:, 1:2]
    W = priors[:, 2:3] * tf.exp(predictions[:, 2:3] * variances[2])
    H = priors[:, 3:4] * tf.exp(predictions[:, 3:4] * variances[3])
    boxes = tf.concat([center_x, center_y, W, H], axis=1)
    boxes = to_corner_form(boxes)
    return tf.concat([boxes, predictions[:, 4:]], 1)


def denormalize_box(box, image_shape):
    """Scales corner box coordinates from normalized values to image
     dimensions.

    # Arguments
        box: Numpy array containing corner box coordinates.
        image_shape: List of integers with (height, width).

    # Returns
        returns: box corner coordinates in image dimensions
    """
    height, width = image_shape
    vals = tf.constant([width, height, width, height], dtype=tf.float32)
    box = tf.multiply(box, vals)
    return box


def scale_box(box, image_scale):
    vals = tf.constant([image_scale[1], image_scale[0],
                        image_scale[1], image_scale[0]], dtype=tf.float32)
    box = tf.multiply(box, vals)
    return box


def ssd_convert_coordinates(conv_outs, prior_boxes, visualize_index,
                            image_size, image_scale):
    conv_outs = decode(conv_outs, prior_boxes)
    conv_outs = conv_outs[visualize_index[1], :4]
    # Anchor boxes are in the normalized image coordinates level
    # conv_outs = denormalize_box(conv_outs, (image_size, image_size))
    # conv_outs = scale_box(conv_outs, image_scale)
    return conv_outs