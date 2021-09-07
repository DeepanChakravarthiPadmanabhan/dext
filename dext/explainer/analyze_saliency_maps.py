import cv2
import numpy as np
from dext.explainer.utils import resize_boxes
from dext.explainer.utils import get_saliency_mask

def analyze_saliency_maps(detections, image, saliency_list, visualize_object_index):
    # include stats such as num detections, iou for each detection on a saliency map,
    boxes = resize_boxes(detections, image.shape, saliency_list[0].shape)
    box = boxes[visualize_object_index - 1]
    iou_compiled = []
    for n, i in enumerate(saliency_list):
        mask_2d = get_saliency_mask(i)
        length = box[2] - box[0] + 1
        height = box[3] - box[1] + 1
        area = length * height
        image_mask = np.zeros((saliency_list[0].shape[0],
                               saliency_list[0].shape[1]))
        pts = np.array([[[box[0], box[1]],
                         [box[0], box[3]],
                         [box[2], box[3]],
                         [box[2], box[1]]]], dtype=np.int32)
        cv2.fillPoly(image_mask, pts, 1)
        white_pixels = image_mask * mask_2d
        num_whites = len(np.where(white_pixels == 1)[0])
        iou = num_whites / area
        iou_compiled.append(iou)
    print("Length of IOU compiled for all saliency maps: ", len(iou_compiled), iou_compiled)


