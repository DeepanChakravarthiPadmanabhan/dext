import os
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

from dext.explainer.utils import resize_box
from dext.explainer.utils import get_model
from dext.explainer.utils import get_saliency_mask
from dext.evaluate.utils import get_evaluation_details
from dext.evaluate.coco_evaluation import get_coco_metrics


def calculate_saliency_iou(mask_2d, box):
    length = box[2] - box[0] + 1
    height = box[3] - box[1] + 1
    area = length * height
    image_mask = np.zeros((mask_2d.shape[0],
                           mask_2d.shape[1]))
    pts = np.array([[[box[0], box[1]],
                     [box[0], box[3]],
                     [box[2], box[3]],
                     [box[2], box[1]]]], dtype=np.int32)
    cv2.fillPoly(image_mask, pts, 1)
    white_pixels = image_mask * mask_2d
    num_whites = len(np.where(white_pixels == 1)[0])
    iou = num_whites / area
    return iou


def calculate_centroid(mask_2d):
    M = cv2.moments(mask_2d)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy


def calculate_variance(mask_2d):
    return np.var(mask_2d)


def analyze_saliency_maps(detections, image, saliency_map,
                          visualize_object_index):
    box = detections[visualize_object_index]
    box = resize_box(box, image.shape, saliency_map.shape)
    mask_2d = get_saliency_mask(saliency_map)
    iou = calculate_saliency_iou(mask_2d, box)
    centroid = calculate_centroid(mask_2d)
    variance = calculate_variance(mask_2d)
    return iou, centroid, variance


def get_object_ap_curve(saliency, raw_image, preprocessor_fn,
                        postprocessor_fn, inference_fn, image_size=512,
                        model_name='SSD512', image_index=None,
                        result_file='ap_curve.json'):
    plt.imsave("ap_img.jpg", raw_image)
    model = get_model(model_name)
    image = deepcopy(raw_image)

    # TODO: Get perturbed image. Calculate AP @.5.
    forward_pass_outs = inference_fn(
        model, image, preprocessor_fn,
        postprocessor_fn, image_size)
    detections = forward_pass_outs[1]
    eval_json = []
    all_boxes = get_evaluation_details(detections)
    for i in all_boxes:
        eval_entry = {'image_id': image_index, 'category_id': i[5],
                      'bbox': i[:4], 'score': i[4]}
        eval_json.append(eval_entry)
    try:
        os.remove(result_file)
    except OSError:
        pass
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(eval_json, f, ensure_ascii=False, indent=4)
    ap_50cent = get_coco_metrics(result_file)

    ap_curve = ap_50cent
    return ap_curve
