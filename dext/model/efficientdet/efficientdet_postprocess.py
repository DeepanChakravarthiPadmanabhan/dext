import numpy as np
import paz.processors as pr
from paz.abstract import SequentialProcessor, Box2D
from dext.utils.class_names import get_class_name_efficientdet
from dext.postprocessing.detection_visualization import draw_bounding_boxes


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
    for class_arg in range(0, num_classes):
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
    for class_arg in range(0, num_classes):
        class_detections = boxes[class_arg, :]
        confidence_mask = np.squeeze(class_detections[:, -2] >= conf_thresh)
        confident_class_detections = class_detections[confidence_mask]
        if len(confident_class_detections) == 0:
            continue
        class_name = arg_to_class[class_arg]
        for confident_class_detection in confident_class_detections:
            coordinates = confident_class_detection[:4]
            score = confident_class_detection[4]
            feature_map_position = confident_class_detection[5]
            boxes2D.append(Box2D(coordinates, score, class_name))
            class_map_idx.append([feature_map_position, class_arg, score])
    return boxes2D, class_map_idx


def process_outputs(outputs):
    """
    Merges all feature levels into single tensor and combines box offsets
    and class scores.

    # Arguments
        class_outputs: Tensor, logits for all classes corresponding to the
        features associated with the box coordinates at each feature levels.
        box_outputs: Tensor, box coordinate offsets for the corresponding prior
        boxes at each feature levels.
        num_levels: Int, number of levels considered at efficientnet features.
        num_classes: Int, number of classes in the dataset.

    # Returns
        outputs: Numpy array, Processed outputs by merging the features at
        all levels. Each row corresponds to box coordinate offsets and
        sigmoid of the class logits.
    """
    outputs = outputs[0]
    boxes, classes = outputs[:, :4], outputs[:, 4:]
    s1, s2, s3, s4 = np.hsplit(boxes, 4)
    boxes = np.concatenate([s2, s1, s4, s3], axis=1)
    boxes = boxes[np.newaxis]
    classes = classes[np.newaxis]
    outputs = np.concatenate([boxes, classes], axis=2)
    return outputs


def scale_box(box, image_scale):
    x_min, y_min, x_max, y_max = box[:4]
    x_min = int(x_min * image_scale[1])
    y_min = int(y_min * image_scale[0])
    x_max = int(x_max * image_scale[1])
    y_max = int(y_max * image_scale[0])
    return x_min, y_min, x_max, y_max


def scale_boxes(boxes2d, image_scale):
    for box2d in boxes2d:
        box2d.coordinates = scale_box(box2d.coordinates, image_scale)
    return boxes2d


def efficientdet_postprocess(model, outputs, image_scales, raw_images=None,
                             explain_top5_backgrounds=False):
    outputs = process_outputs(outputs)
    postprocessing = SequentialProcessor([
        pr.Squeeze(axis=None),
        pr.DecodeBoxes(model.prior_boxes, variances=[1, 1, 1, 1])])
    detections = postprocessing(outputs)
    detections = nms_per_class(detections, 0.4)
    detections, class_map_idx = filterboxes(
        detections, get_class_name_efficientdet('COCO'), 0.4)
    detections = scale_boxes(detections, image_scales)
    image = draw_bounding_boxes(raw_images.astype('uint8'),
                                detections,
                                get_class_name_efficientdet('COCO'))

    if explain_top5_backgrounds:
        image, detections, class_map_idx = get_top5_bg_efficientdet(
            model, outputs, image_scales, raw_images)
    return image, detections, class_map_idx


def filterboxes_bg(boxes, class_names, conf_thresh=0.5):
    arg_to_class = dict(zip(list(range(len(class_names))), class_names))
    num_classes = boxes.shape[0]
    boxes2D = []
    class_map_idx = []
    for class_arg in range(0, num_classes):
        class_detections = boxes[class_arg, :]
        confidence_mask = np.squeeze(class_detections[:, -2] < conf_thresh)
        confident_class_detections = class_detections[confidence_mask]
        if len(confident_class_detections) == 0:
            continue
        class_name = arg_to_class[class_arg]
        for confident_class_detection in confident_class_detections:
            coordinates = confident_class_detection[:4]
            if (coordinates[0] >= coordinates[2]) and\
                    (coordinates[1] >= coordinates[3]):
                continue
            score = confident_class_detection[4]
            feature_map_position = confident_class_detection[5]
            boxes2D.append(Box2D(coordinates, score, class_name))
            class_map_idx.append([feature_map_position, class_arg, score])
    return boxes2D, class_map_idx


def select_top5_bg_det(detections, class_map_idx, order='top5'):
    score_list = []
    for box in detections:
        score_list.append(box.score)
    score_list = np.array(score_list)
    score_list_arg = np.argsort(score_list)
    if order == 'top5':
        score_list_arg = score_list_arg[::-1][:5]
    else:
        score_list_arg = score_list_arg[:5]
    detections = np.array(detections)
    class_map_idx_selected = [class_map_idx[i] for i in score_list_arg]
    detections_selected = detections[score_list_arg]
    return detections_selected, class_map_idx_selected


def get_bg_dets(detections, image_scales, raw_images):
    bg_det, class_map_idx = filterboxes_bg(
        detections, get_class_name_efficientdet('COCO'), 0.4)
    bg_det = scale_boxes(bg_det, image_scales)
    bg_det, class_map_idx = select_top5_bg_det(bg_det, class_map_idx, 'top5')
    image = draw_bounding_boxes(raw_images.astype('uint8'), bg_det,
                                get_class_name_efficientdet('COCO'))
    return image, bg_det, class_map_idx


def get_top5_bg_efficientdet(model, outputs, image_scales, raw_image=None):
    outputs = process_outputs(outputs)
    postprocessing = SequentialProcessor([
        pr.Squeeze(axis=None),
        pr.DecodeBoxes(model.prior_boxes, variances=[1, 1, 1, 1])])
    detections = postprocessing(outputs)
    detections = nms_per_class(detections, 0.4)
    image, bg_det, class_map_idx = get_bg_dets(detections, image_scales,
                                               raw_image)
    return image, bg_det, class_map_idx
