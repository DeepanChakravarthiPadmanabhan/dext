from paz.datasets.utils import get_class_names

coco_efficientdet = ['person', 'bicycle', 'car', 'motorcycle',
                     'airplane', 'bus', 'train', 'truck', 'boat',
                     'traffic light', 'fire hydrant', '0', 'stop sign',
                     'parking meter', 'bench', 'bird', 'cat', 'dog',
                     'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                     'giraffe', '0', 'backpack', 'umbrella', '0',
                     '0', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                     'snowboard', 'sports ball', 'kite', 'baseball bat',
                     'baseball glove', 'skateboard', 'surfboard',
                     'tennis racket', 'bottle', '0', 'wine glass', 'cup',
                     'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                     'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                     'pizza', 'donut', 'cake', 'chair', 'couch',
                     'potted plant', 'bed', '0', 'dining table', '0', '0',
                     'toilet', '0', 'tv', 'laptop', 'mouse', 'remote',
                     'keyboard', 'cell phone', 'microwave', 'oven',
                     'toaster', 'sink', 'refrigerator', '0', 'book',
                     'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                     'toothbrush']


coco_to_voc = {'airplane': 'aeroplane',
               'motorcycle': 'motorbike',
               'dining table': 'diningtable',
               'potted plant': 'pottedplant',
               'tv': 'tvmonitor',
               'couch': 'sofa'}


voc_to_coco = {'aeroplane': 'airplane',
               'motorbike': 'motorcycle',
               'diningtable': 'dining table',
               'pottedplant': 'potted plant',
               'tvmonitor': 'tv',
               'sofa': 'couch'}


voc_class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                   'diningtable', 'dog', 'horse', 'motorbike', 'person',
                   'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


coco_fasterrcnn = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench',
                   'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
                   'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                   'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                   'snowboard', 'sports ball', 'kite', 'baseball bat',
                   'baseball glove', 'skateboard', 'surfboard',
                   'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                   'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                   'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                   'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']


def get_classes(dataset_name, model_name):
    if dataset_name == 'COCO' and 'EFFICIENTDET' in model_name:
        return coco_efficientdet
    elif dataset_name == 'COCO' and model_name == 'FasterRCNN':
        return coco_fasterrcnn
    elif dataset_name == 'COCO' and model_name == 'SSD512':
        return get_class_names(dataset_name)
    elif dataset_name == 'VOC':
        return voc_class_names
    else:
        raise ValueError('Dataset and model combination unavailable.')
