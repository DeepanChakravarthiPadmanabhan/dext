import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

from dext.model.faster_rcnn.layer_utils import batch_slice
from dext.model.faster_rcnn.layer_utils import (
    clip_boxes_graph, apply_box_deltas_graph)


class DetectionLayer(Layer):
    """Detects final bounding boxes and masks for given proposals

    # Arguments:
        config: instance of base configuration class

    # Returns:
        [batch, num_detections, (y_min, x_min, y_max, x_max, class_id,
         class_score)] where coordinates are normalized.
    """

    def __init__(self, batch_size, window, bbox_std_dev, images_per_gpu,
                 detection_max_instances, detection_min_confidence,
                 detection_nms_threshold, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.BATCH_SIZE = batch_size
        self.WINDOW = window
        self.BBOX_STD_DEV = bbox_std_dev
        self.IMAGES_PER_GPU = images_per_gpu
        self.DETECTION_MAX_INSTANCES = detection_max_instances
        self.DETECTION_MIN_CONFIDENCE = detection_min_confidence
        self.DETECTION_NMS_THRESHOLD = detection_nms_threshold

    def call(self, inputs):
        ROIs, mrcnn_class, mrcnn_bbox = inputs
        detections_batch = batch_slice(
            [ROIs, mrcnn_class, mrcnn_bbox],
            lambda x, y, w: self.refine_detections_graph(x, y, w),
            self.IMAGES_PER_GPU)
        return tf.reshape(detections_batch,
                          [self.BATCH_SIZE, self.DETECTION_MAX_INSTANCES, 6])

    def refine_detections_graph(self, rois, probs, deltas):
        class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
        indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
        class_scores = tf.gather_nd(probs, indices)
        deltas_specific = tf.gather_nd(deltas, indices)

        refined_rois = apply_box_deltas_graph(
            rois, deltas_specific * self.BBOX_STD_DEV)
        refined_rois = clip_boxes_graph(refined_rois, self.WINDOW)
        keep = tf.where(class_ids > 0)[:, 0]

        if self.DETECTION_MIN_CONFIDENCE:
            keep = self.filter_low_confidence(class_scores, keep)

        nms_keep = self.apply_NMS(class_ids, class_scores, refined_rois, keep)
        keep = self.get_top_detections(class_scores, keep, nms_keep)

        detections = tf.concat([
            tf.gather(refined_rois, keep),
            tf.cast(tf.gather(class_ids, keep),
                    dtype=tf.float32)[..., tf.newaxis],
            tf.gather(class_scores, keep)[..., tf.newaxis]], axis=1)

        return self.zero_pad_detections(detections)

    def get_top_detections(self, scores, keep, nms_keep):
        keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
        keep = tf.sparse.to_dense(keep)[0]
        roi_count = self.DETECTION_MAX_INSTANCES
        class_scores_keep = tf.gather(scores, keep)
        num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
        top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
        return tf.gather(keep, top_ids)

    def NMS_map(self, pre_nms_elements, keep, unique_class_id):
        class_ids, scores, ROIs = pre_nms_elements
        ixs = tf.where(tf.equal(class_ids, unique_class_id))[:, 0]

        class_keep = tf.image.non_max_suppression(
            tf.gather(ROIs, ixs),
            tf.gather(scores, ixs),
            max_output_size=self.DETECTION_MAX_INSTANCES,
            iou_threshold=self.DETECTION_NMS_THRESHOLD)
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        gap = self.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)],
                            mode='CONSTANT', constant_values=-1)
        class_keep.set_shape([self.DETECTION_MAX_INSTANCES])

        return class_keep

    def apply_NMS(self, class_ids, scores, refined_rois, keep):
        pre_nms_class_ids = tf.gather(class_ids, keep)
        pre_nms_scores = tf.gather(scores, keep)
        pre_nms_rois = tf.gather(refined_rois, keep)
        unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

        pre_nms_elements = [pre_nms_class_ids, pre_nms_scores, pre_nms_rois]
        nms_keep = tf.map_fn(lambda x: self.NMS_map(pre_nms_elements, keep, x),
                             unique_pre_nms_class_ids,
                             dtype=tf.int64)

        return self.merge_results(nms_keep)

    def filter_low_confidence(self, class_scores, keep):
        confidence = tf.where(
            class_scores >= self.DETECTION_MIN_CONFIDENCE)[:, 0]
        keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(confidence, 0))
        return tf.sparse.to_dense(keep)[0]

    def merge_results(self, nms_keep):
        nms_keep = tf.reshape(nms_keep, [-1])
        return tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])

    def zero_pad_detections(self, detections):
        gap = self.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
        return tf.pad(detections, [(0, gap), (0, 0)], 'CONSTANT')

    def compute_output_shape(self, input_shape):
        return (None, self.DETECTION_MAX_INSTANCES, 6)


class ProposalLayer(Layer):
    """Receives anchor scores and selects a subset to pass as proposals
       to the second stage. Filtering is done based on anchor scores and
       non-max suppression to remove overlaps. It also applies bounding
       box refinement deltas to anchors.

    # Arguments:
        rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, num_anchors, (y_min, x_min, y_max, x_max)] anchors
                 in normalized coordinates

    # Returns:
        Normalized proposals [batch, rois, (y_min, x_min, y_max, x_max)]
    """

    def __init__(self, proposal_count, nms_threshold, rpn_bbox_std_dev,
                 pre_nms_limit, images_per_gpu, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.RPN_BBOX_STD_DEV = rpn_bbox_std_dev
        self.PRE_NMS_LIMIT = pre_nms_limit
        self.IMAGES_PER_GPU = images_per_gpu
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def call(self, inputs):
        scores, deltas, anchors = inputs
        scores = scores[:, :, 1]
        deltas = deltas * np.reshape(self.RPN_BBOX_STD_DEV, [1, 1, 4])

        scores, deltas, pre_nms_anchors = self.trim_by_score(scores, deltas,
                                                             anchors)
        boxes = self.apply_box_delta(pre_nms_anchors, deltas)
        boxes = self.clip_image_boundaries(boxes)

        proposals = batch_slice([boxes, scores], self.NMS,
                                self.IMAGES_PER_GPU)
        return proposals

    def NMS(self, boxes, scores):
        indices = tf.image.non_max_suppression(
            boxes, scores, self.proposal_count,
            self.nms_threshold, name='rpn_non_max_suppression')
        proposals = tf.gather(boxes, indices)
        padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
        proposals = tf.pad(proposals, [(0, padding), (0, 0)])
        return proposals

    def trim_by_score(self, scores, deltas, anchors):
        pre_nms_limit = tf.minimum(self.PRE_NMS_LIMIT, tf.shape(anchors)[1])
        indices = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                              name='top_anchors').indices
        scores = batch_slice([scores, indices], lambda x, y: tf.gather(x, y),
                             self.IMAGES_PER_GPU)
        deltas = batch_slice([deltas, indices], lambda x, y: tf.gather(x, y),
                             self.IMAGES_PER_GPU)
        pre_nms_anchors = batch_slice([anchors, indices],
                                      lambda a, x: tf.gather(a, x),
                                      self.IMAGES_PER_GPU,
                                      names=['pre_nms_anchors'])
        return scores, deltas, pre_nms_anchors

    def apply_box_delta(self, pre_nms_anchors, deltas):
        boxes = batch_slice([pre_nms_anchors, deltas],
                            lambda x, y: apply_box_deltas_graph(x, y),
                            self.IMAGES_PER_GPU,
                            names=['refined_anchors'])
        return boxes

    def clip_image_boundaries(self, boxes):
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = batch_slice(boxes, lambda x: clip_boxes_graph(x, window),
                            self.IMAGES_PER_GPU,
                            names=['refined_anchors_clipped'])
        return boxes

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)


class PyramidROIAlign(Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    # Arguments:
        pool_shape: [pool_height, pool_width] of the output pooled regions
        boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
                coordinates. Possibly padded with zeros if not enough
                boxes to fill the array.
        image_shape: shape of image
        feature_maps: List of feature maps from different levels
                      of the pyramid. Each is [batch, height, width, channels]

    # Returns:
        Pooled regions: [batch, num_boxes, pool_height, pool_width, channels].
        The width and height are those specific in the pool_shape in the layer
        constructor.
    """

    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):
        boxes, image_shape = inputs[0], inputs[1]
        feature_maps = inputs[2:]

        ROI_level = self.compute_ROI_level(boxes, image_shape)
        pooled, box_to_level = self.apply_ROI_pooling(ROI_level, boxes,
                                                      feature_maps)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)
        pooled = self.rearrange_pooled_features(pooled, box_to_level, boxes)
        return pooled

    def compute_ROI_level(self, boxes, image_shape):
        y_min, x_min, y_max, x_max = tf.split(boxes, 4, axis=2)
        height = y_max - y_min
        width = x_max - x_min
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)

        ROI_level = self.log2_graph(
            tf.sqrt(height * width) / (224.0 / tf.sqrt(image_area)))
        ROI_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(ROI_level), tf.int32)))
        return tf.squeeze(ROI_level, 2)

    def apply_ROI_pooling(self, ROI_level, boxes, feature_maps):
        pooled, box_to_level = [], []
        for index, level in enumerate(range(2, 6)):
            level_index = tf.where(tf.equal(ROI_level, level))
            level_boxes = tf.gather_nd(boxes, level_index)
            box_indices = tf.cast(level_index[:, 0], tf.int32)
            box_to_level.append(level_index)

            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)
            pooled.append(tf.image.crop_and_resize(
                feature_maps[index], level_boxes, box_indices, self.pool_shape,
                method='bilinear'))
        pooled = tf.concat(pooled, axis=0)
        box_to_level = tf.concat(box_to_level, axis=0)
        return pooled, box_to_level

    def rearrange_pooled_features(self, pooled, box_to_level, boxes):
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        top_k_indices = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        top_k_indices = tf.gather(box_to_level[:, 2], top_k_indices)
        pooled = tf.gather(pooled, top_k_indices)
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        return tf.reshape(pooled, shape)

    def log2_graph(self, x):
        return tf.math.log(x) / tf.math.log(2.0)

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1],)
