import os
import logging
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

from paz.backend.image.opencv_image import resize_image
from dext.explainer.utils import resize_box
from dext.explainer.utils import get_model
from dext.explainer.utils import get_saliency_mask
from dext.evaluate.utils import get_evaluation_details
from dext.evaluate.coco_evaluation import get_coco_metrics

LOGGER = logging.getLogger(__name__)


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


def save_modified_image(raw_image, name, saliency_shape, change_pixels):
    image = resize_image(raw_image, saliency_shape)
    for pix in change_pixels:
        image[pix[0], pix[1], :] = 0
    modified_image = resize_image(image,
                                  (raw_image.shape[1], raw_image.shape[0]))
    plt.imsave('modified_image' + str(name) + '.jpg',
               modified_image.astype('uint8'))


def get_object_ap_curve(saliency, raw_image, preprocessor_fn, postprocessor_fn,
                        image_size=512, model_name='SSD512', image_index=None,
                        result_file='ap_curve.json',
                        save_modified_images=False):
    model = get_model(model_name)
    num_pixels = saliency.size
    percentage_space = np.linspace(0, 1, 10)
    sorted_saliency = (-saliency).argsort(axis=None, kind='mergesort')
    sorted_flat_indices = np.unravel_index(sorted_saliency, saliency.shape)
    sorted_indices = np.vstack(sorted_flat_indices).T
    ap_curve = []
    input_image = deepcopy(raw_image)
    for n, percent in enumerate(percentage_space):
        modified_image, image_scales = preprocessor_fn(input_image, image_size)
        num_pixels_selected = int(num_pixels * percent)
        change_pixels = sorted_indices[:num_pixels_selected]
        modified_image = modified_image[0]
        for pix in change_pixels:
            modified_image[pix[0], pix[1], :] = 0
        modified_image = modified_image[np.newaxis]
        outputs = model(modified_image)
        detection_image, detections, _ = postprocessor_fn(
            model, outputs, image_scales, raw_image)
        if save_modified_images:
            save_modified_image(input_image, n, saliency.shape, change_pixels)
            plt.imsave("detection_image" + str(n) + '.jpg', detection_image)
        if len(detections):
            eval_json = []
            all_boxes = get_evaluation_details(detections)
            for box in all_boxes:
                eval_entry = {'image_id': image_index, 'category_id': box[4],
                              'bbox': box[:4], 'score': box[5]}
                eval_json.append(eval_entry)
            try:
                os.remove(result_file)
            except OSError:
                pass
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(eval_json, f, ensure_ascii=False, indent=4)
            from dext.utils.constants import COCO_VAL_ANNOTATION_FILE
            ap_50cent = get_coco_metrics(result_file, COCO_VAL_ANNOTATION_FILE)
            LOGGER.info('AP 50 at modification percentage %s: %s' % (
                percent, ap_50cent))
            ap_curve.append(ap_50cent)
        else:
            LOGGER.info('No detections. Mapping AP to 0.')
            ap_curve.append(0)
    LOGGER.info('AP curve: %s' % ap_curve)
    return ap_curve
