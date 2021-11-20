import numpy as np

from paz.abstract import Box2D
from dext.model.faster_rcnn.utils import norm_boxes
from dext.model.faster_rcnn.utils import denorm_boxes
from dext.postprocessing.detection_visualization import draw_bounding_boxes
from dext.model.faster_rcnn.layer_utils import apply_box_deltas_graph
from dext.model.faster_rcnn.layer_utils import clip_boxes_graph
from dext.model.faster_rcnn.utils import norm_boxes_graph


class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


def apply_non_max_suppression(boxes, scores, iou_thresh=.45, top_k=200):
    """Apply non maximum suppression.

    # Arguments
        boxes: Numpy array, box coordinates of shape `(num_boxes, 4)`
            where each columns corresponds to x_min, y_min, x_max, y_max.
        scores: Numpy array, of scores given for each box in `boxes`.
        iou_thresh: float, intersection over union threshold for removing
            boxes.
        top_k: int, number of maximum objects per class.

    # Returns
        selected_indices: Numpy array, selected indices of kept boxes.
        num_selected_boxes: int, number of selected boxes.
    """

    selected_indices = np.zeros(shape=len(scores))
    if boxes is None or len(boxes) == 0:
        return selected_indices
    x_min = boxes[:, 0]
    y_min = boxes[:, 1]
    x_max = boxes[:, 2]
    y_max = boxes[:, 3]
    areas = (x_max - x_min) * (y_max - y_min)
    remaining_sorted_box_indices = np.argsort(scores)
    remaining_sorted_box_indices = remaining_sorted_box_indices[-top_k:]

    num_selected_boxes = 0
    while len(remaining_sorted_box_indices) > 0:
        best_score_args = remaining_sorted_box_indices[-1]
        selected_indices[num_selected_boxes] = best_score_args
        num_selected_boxes = num_selected_boxes + 1
        if len(remaining_sorted_box_indices) == 1:
            break

        remaining_sorted_box_indices = remaining_sorted_box_indices[:-1]

        best_x_min = x_min[best_score_args]
        best_y_min = y_min[best_score_args]
        best_x_max = x_max[best_score_args]
        best_y_max = y_max[best_score_args]

        remaining_x_min = x_min[remaining_sorted_box_indices]
        remaining_y_min = y_min[remaining_sorted_box_indices]
        remaining_x_max = x_max[remaining_sorted_box_indices]
        remaining_y_max = y_max[remaining_sorted_box_indices]

        inner_x_min = np.maximum(remaining_x_min, best_x_min)
        inner_y_min = np.maximum(remaining_y_min, best_y_min)
        inner_x_max = np.minimum(remaining_x_max, best_x_max)
        inner_y_max = np.minimum(remaining_y_max, best_y_max)

        inner_box_widths = inner_x_max - inner_x_min
        inner_box_heights = inner_y_max - inner_y_min

        inner_box_widths = np.maximum(inner_box_widths, 0.0)
        inner_box_heights = np.maximum(inner_box_heights, 0.0)

        intersections = inner_box_widths * inner_box_heights
        remaining_box_areas = areas[remaining_sorted_box_indices]
        best_area = areas[best_score_args]
        unions = remaining_box_areas + best_area - intersections
        intersec_over_union = intersections / unions
        intersec_over_union_mask = intersec_over_union <= iou_thresh
        remaining_sorted_box_indices = remaining_sorted_box_indices[
            intersec_over_union_mask]

    return selected_indices.astype(int), num_selected_boxes


def nms_per_class(box_data, nms_thresh=0.45, conf_thresh=0.01, top_k=200):
    decoded_boxes, class_predictions = box_data[:, :4], box_data[:, 4:]
    num_classes = class_predictions.shape[1]
    output = np.zeros((num_classes, top_k, 6))
    # skip the background class (start counter in 1)
    for class_arg in range(1, num_classes):
        conf_mask = class_predictions[:, class_arg] >= conf_thresh
        scores = class_predictions[:, class_arg][conf_mask]
        if len(scores) == 0:
            continue
        boxes = decoded_boxes[conf_mask]
        boxes_raw_index = np.where(conf_mask)[0]
        indices, count = apply_non_max_suppression(
            boxes, scores, nms_thresh, top_k)
        scores = np.expand_dims(scores, -1)
        selected_indices = indices[:count]
        select_boxes_raw = boxes_raw_index[indices[:count]]
        selections = np.concatenate(
            (boxes[selected_indices], scores[selected_indices],
             np.expand_dims(select_boxes_raw, 1)), axis=1)
        output[class_arg, :count, :] = selections
    return output


def filterboxes(boxes, class_names, conf_thresh=0.5):
    arg_to_class = dict(zip(list(range(len(class_names))), class_names))
    num_classes = boxes.shape[0]
    boxes2D = []
    class_map_idx = []
    for class_arg in range(1, num_classes):
        class_detections = boxes[class_arg, :]
        confidence_mask = np.squeeze(class_detections[:, -2] >= conf_thresh)
        confident_class_detections = class_detections[confidence_mask]
        if len(confident_class_detections) == 0:
            continue
        class_name = arg_to_class[class_arg]
        for confident_class_detection in confident_class_detections:
            coordinates = confident_class_detection[:4]
            x_min, y_min = coordinates[1], coordinates[0]
            x_max, y_max = coordinates[3], coordinates[2]
            coordinates = [x_min, y_min, x_max, y_max]
            score = confident_class_detection[4]
            feature_map_position = confident_class_detection[5]
            boxes2D.append(Box2D(coordinates, score, class_name))
            class_map_idx.append([feature_map_position, class_arg, score])
    return boxes2D, class_map_idx


def faster_rcnn_postprocess(model, outputs, image_scales, image,
                            image_size=512, explain_top5_background=False):
    outputs = outputs[0]
    rois = outputs[:, 85:]
    offsets = outputs[:, :4]
    class_confidences = outputs[:, 4:-4]
    refined_rois = apply_box_deltas_graph(
        rois, offsets * np.array([0.1, 0.1, 0.2, 0.2]))
    config_window = norm_boxes_graph(image_scales, (image_size, image_size))
    refined_rois = clip_boxes_graph(refined_rois, config_window)
    detections = np.concatenate([refined_rois, class_confidences], axis=1)
    scaled_boxes = scale_boxes(detections[:, :4], image_scales,
                               (image_size, image_size, 3), image.shape)
    detections = np.concatenate([scaled_boxes, detections[:, 4:]], axis=1)
    detections = nms_per_class(detections, 0.3)
    detections, class_map_idx = filterboxes(detections, class_names, 0.7)
    image = draw_bounding_boxes(image, detections, class_names,
                                max_size=image_size)
    return image, detections, class_map_idx


def scale_boxes(boxes, window, image_shape, original_image_shape):
    window = norm_boxes(window, image_shape[:2])
    Wy_min, Wx_min, Wy_max, Wx_max = window
    shift = np.array([Wy_min, Wx_min, Wy_min, Wx_min])
    window_H = Wy_max - Wy_min
    window_W = Wx_max - Wx_min
    scale = np.array([window_H, window_W, window_H, window_W])
    boxes = np.divide(boxes - shift, scale)
    boxes = denorm_boxes(boxes, original_image_shape[:2])
    return boxes
