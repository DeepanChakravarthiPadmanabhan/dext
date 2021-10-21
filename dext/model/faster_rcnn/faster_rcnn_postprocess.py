import numpy as np

from dext.model.faster_rcnn.utils import norm_boxes
from dext.model.faster_rcnn.utils import denorm_boxes

from paz.abstract import Box2D
from dext.postprocessing.detection_visualization import draw_bounding_boxes


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


def get_box_index_faster_rcnn(outputs, num_detections):
    box_index = []
    for i in range(num_detections):
        box_index.append([i, int(outputs[i, 4]), outputs[i, 5]])
    return box_index


def faster_rcnn_postprocess(model, outputs, image_scales, image,
                            image_size=512, explain_top5_background=False):
    outputs = outputs.numpy()[0]
    boxes, class_ids, scores = postprocess(
        outputs, image.shape, (image_size, image_size, 3),
        image_scales)
    detections = []
    for box, score, class_name in zip(boxes, scores, class_ids):
        x_min, y_min, x_max, y_max = box[1], box[0], box[3], box[2]
        detections.append(Box2D([x_min, y_min, x_max, y_max], score,
                                class_names[class_name]))
    image = draw_bounding_boxes(image.astype('uint8'), detections, class_names)
    box_index = get_box_index_faster_rcnn(outputs, len(detections))
    return image, detections, box_index


def postprocess(detections, original_image_shape,
                image_shape, window):
    zero_index = np.where(detections[:, 4] == 0)[0]
    N = zero_index[0] if zero_index.shape[0] > 0 else detections.shape[0]
    boxes, class_ids, scores = unpack_detections(N, detections)
    boxes = normalize_boxes(boxes, window, image_shape, original_image_shape)
    boxes, class_ids, scores, N = filter_detections(
        N, boxes, class_ids, scores)
    return boxes, class_ids, scores


def unpack_detections(N, detections):
    boxes = detections[:N, :4]
    class_ids = detections[:N, 4].astype(np.int32)
    scores = detections[:N, 5]
    return boxes, class_ids, scores


def normalize_boxes(boxes, window, image_shape,
                    original_image_shape):
    window = norm_boxes(window, image_shape[:2])
    Wy_min, Wx_min, Wy_max, Wx_max = window
    shift = np.array([Wy_min, Wx_min, Wy_min, Wx_min])
    window_H = Wy_max - Wy_min
    window_W = Wx_max - Wx_min
    scale = np.array([window_H, window_W, window_H, window_W])
    boxes = np.divide(boxes - shift, scale)
    boxes = denorm_boxes(boxes, original_image_shape[:2])
    return boxes


def filter_detections(N, boxes, class_ids, scores):
    exclude_index = np.where(
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
    if exclude_index.shape[0] > 0:
        boxes = np.delete(boxes, exclude_index, axis=0)
        class_ids = np.delete(class_ids, exclude_index, axis=0)
        scores = np.delete(scores, exclude_index, axis=0)
        N = class_ids.shape[0]
    return boxes, class_ids, scores, N
