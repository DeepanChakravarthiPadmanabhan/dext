coco_class_names = ['person', 'bicycle', 'car', 'motorcycle',
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


def get_class_name_efficientdet(dataset_name):
    if dataset_name == 'COCO':
        return coco_class_names
