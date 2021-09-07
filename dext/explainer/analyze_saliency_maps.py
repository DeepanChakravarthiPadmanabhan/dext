import cv2
import numpy as np


from dext.explainer.utils import resize_box
from dext.explainer.utils import get_saliency_mask

def analyze_saliency_maps(detections, image, saliency_map, visualize_object_index):
    # include stats such as num detections, iou for each detection on a saliency map,
    box = detections[visualize_object_index]
    box = resize_box(box, image.shape, saliency_map.shape)
    mask_2d = get_saliency_mask(saliency_map)
    length = box[2] - box[0] + 1
    height = box[3] - box[1] + 1
    area = length * height
    image_mask = np.zeros((saliency_map.shape[0],
                           saliency_map.shape[1]))
    pts = np.array([[[box[0], box[1]],
                     [box[0], box[3]],
                     [box[2], box[3]],
                     [box[2], box[1]]]], dtype=np.int32)
    cv2.fillPoly(image_mask, pts, 1)
    white_pixels = image_mask * mask_2d
    num_whites = len(np.where(white_pixels == 1)[0])
    iou = num_whites / area
    return iou


