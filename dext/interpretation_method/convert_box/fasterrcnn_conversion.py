import tensorflow as tf
from dext.model.faster_rcnn.utils import norm_boxes_graph
from dext.model.faster_rcnn.layer_utils import apply_box_deltas_graph
from dext.model.faster_rcnn.layer_utils import clip_boxes_graph


def norm_boxes(boxes, shape):
    h, w = shape
    scale = tf.constant([h - 1, w - 1, h - 1, w - 1])
    shift = tf.constant([0, 0, 1, 1])
    return tf.math.divide((boxes - shift), scale)


def denorm_boxes(boxes, shape):
    h, w = shape
    scale = tf.constant([h - 1, w - 1, h - 1, w - 1], tf.float32)
    shift = tf.constant([0, 0, 1, 1], dtype=tf.float32)
    return tf.multiply(boxes, scale) + shift


def scale_boxes(boxes, window, original_image_shape, to_ic):
    """Translate normalized coordinates in the resized image to pixel
    coordinates in the original image."""
    # Window - normalized window on the resized image size
    Wy_min, Wx_min, Wy_max, Wx_max = window
    shift = tf.concat([Wy_min, Wx_min, Wy_min, Wx_min], axis=0)
    shift = tf.expand_dims(shift, axis=0)
    shift = tf.cast(shift, dtype=tf.float32)
    window_H = Wy_max - Wy_min
    window_W = Wx_max - Wx_min
    scale = tf.concat([window_H, window_W, window_H, window_W], axis=0)
    scale = tf.expand_dims(scale, axis=0)
    scale = tf.cast(scale, dtype=tf.float32)
    # Convert boxes to normalized coordinates on the window in original image
    boxes = tf.math.divide(boxes - shift, scale)
    # Convert boxes to pixel coordinates on the original image
    if to_ic:
        boxes = denorm_boxes(boxes, original_image_shape[:2])
    return boxes


def process_outputs(outputs):
    boxes = outputs[:, :4]
    s1, s2, s3, s4 = tf.split(boxes, num_or_size_splits=4, axis=1)
    boxes = tf.concat([s2, s1, s4, s3], axis=1)
    return boxes


def get_fasterrcnn_boxes(conv_outs, original_image_shape, image_size,
                         image_scale, to_ic):
    rois = conv_outs[:, 85:]
    offsets = conv_outs[:, :4]
    # class_confidences = conv_outs[:, 4:-4]
    refined_rois = apply_box_deltas_graph(
        rois, offsets * tf.constant([0.1, 0.1, 0.2, 0.2]))
    # Converts image box from the pixel coordinates to normalized coordinates
    # image scales here is the max box of the input image preserving the AR
    # Normalized window of image on the resized image size.
    config_window = norm_boxes_graph(image_scale, (image_size, image_size))
    # Clip ROI boxes to the resized window size
    refined_rois = clip_boxes_graph(refined_rois, config_window)
    # detections = tf.concat([refined_rois, class_confidences], axis=1)
    conv_outs = scale_boxes(refined_rois, config_window, original_image_shape,
                            to_ic)
    conv_outs = process_outputs(conv_outs)
    return conv_outs


def fasterrcnn_convert_coordinates(conv_outs, original_image_shape,
                                   visualize_idx, image_size, image_scale,
                                   to_ic=False):
    """Converts the detection of one image from the format of the neural
    network output to a format suitable for use in the rest of the
    application. """
    conv_outs = get_fasterrcnn_boxes(conv_outs, original_image_shape,
                                     image_size, image_scale, to_ic)
    conv_outs = conv_outs[visualize_idx[1], :4]
    return conv_outs
