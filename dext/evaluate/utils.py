def get_category_id(class_name):
    coco_id_name_map = {
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
        6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
        11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
        16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
        22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
        28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
        34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball',
        38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard',
        42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass',
        47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl',
        52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
        56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
        61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
        67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
        75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave',
        79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
        85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
        89: 'hair drier', 90: 'toothbrush'}
    names = list(coco_id_name_map.values())
    ids = list(coco_id_name_map.keys())
    select = names.index(class_name)
    return ids[select]


def get_evaluation_box(box, box_format):
    """Scales corner box coordinates from normalized values to image dimensions.

    # Arguments
        box: paz datatype Box.

    # Returns
        returns: List
    """
    x_min, y_min, x_max, y_max = box.coordinates[:4]
    class_id = get_category_id(box.class_name)
    if box_format == 'coco':
        w = float(x_max - x_min)
        h = float(y_max - y_min)
        return [float(x_min), float(y_min), w, h, class_id, box.score]
    else:
        return [float(x_min), float(y_min), float(x_max), float(y_max),
                class_id, box.score]


def get_evaluation_details(boxes2d, box_format='coco'):
    all_boxes = []
    for box2d in boxes2d:
        all_boxes.append(get_evaluation_box(box2d, box_format))
    return all_boxes
