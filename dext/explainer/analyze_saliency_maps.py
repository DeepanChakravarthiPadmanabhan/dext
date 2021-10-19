import os
import logging
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import gin

from paz.backend.image.opencv_image import resize_image
from dext.explainer.utils import resize_box
from dext.explainer.utils import get_model
from dext.explainer.utils import get_saliency_mask
from dext.evaluate.utils import get_evaluation_details
from dext.evaluate.coco_evaluation import get_coco_metrics
from paz.backend.boxes import compute_iou
from dext.evaluate.utils import get_category_id

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


def get_regerror(interest_det, interest_gt):
    det_box = interest_det[:4]
    gt_box = interest_gt[:4]
    min_distance = np.sqrt((det_box[0] - gt_box[0]) ** 2 +
                           (det_box[1] - gt_box[1]) ** 2)
    max_distance = np.sqrt((det_box[2] - gt_box[2]) ** 2 +
                           (det_box[3] - gt_box[3]) ** 2)
    total_distance_error = min_distance + max_distance
    return total_distance_error


def get_flipstatus_maxprob_regerror(all_det_boxes, interest_gt, percent):
    iou = compute_iou(interest_gt[:4], np.array(all_det_boxes)[:, :4])
    max_arg = np.argmax(iou)
    interest_gt_class = int(interest_gt[4])
    interest_det = all_det_boxes[max_arg]
    interest_det_class = int(interest_det[4])
    class_match = interest_gt_class == interest_det_class

    if class_match and iou[max_arg] > 0.4:
        max_prob = all_det_boxes[max_arg][-1]
        reg_error = get_regerror(interest_det, interest_gt)
        LOGGER.info('No flip, IoU: %s, Maxprob: %s, Regerror: %s, '
                    'GT: %s, DET: %s' % (iou, max_prob, reg_error,
                                         interest_gt, interest_det))
        return False, max_prob, reg_error
    else:
        max_prob = 0
        reg_error = np.inf
        LOGGER.info('Flip, IoU: %s, Maxprob: %s, Regerror: %s, '
                    'GT: %s, DET: %s' % (iou, max_prob, reg_error,
                                         interest_gt, interest_det))
        return True, max_prob, reg_error


def get_interest_gt(interest_det, gt_boxes):
    iou = compute_iou(interest_det.coordinates[:4], np.array(gt_boxes)[:, :4])
    max_arg = np.argmax(iou)
    det_class_id = get_category_id(interest_det.class_name)
    if int(gt_boxes[max_arg][4] == int(det_class_id)) and iou[max_arg] > 0.4:
        return int(max_arg)
    else:
        return None


def get_interest_det(detection):
    det_interest_box = list(detection.coordinates[:4])
    det_interest_score = detection.score
    det_interest_class = int(get_category_id(detection.class_name))
    det_interest = det_interest_box
    det_interest += [det_interest_class, det_interest_score]
    return det_interest


def eval_numflip_maxprob_regerror(
        saliency, raw_image, gt_boxes, detections, preprocessor_fn,
        postprocessor_fn, image_size=512, model_name='EFFICIENTDETD0',
        object_index=None, ap_curve_linspace=10,
        explain_top5_backgrounds=False, save_modified_images=True):
    det_matching_interest_det = get_interest_det(detections[object_index])
    LOGGER.info('Evaluating numflip, maxprob, regerror on detection: %s'
                % det_matching_interest_det)
    num_pixels = saliency.size
    percentage_space = np.linspace(0, 1, ap_curve_linspace)
    sorted_saliency = (-saliency).argsort(axis=None, kind='mergesort')
    sorted_flat_indices = np.unravel_index(sorted_saliency, saliency.shape)
    sorted_indices = np.vstack(sorted_flat_indices).T
    input_image = deepcopy(raw_image)
    max_prob_list = [0, ] * len(percentage_space)
    reg_error_list = [np.inf, ] * len(percentage_space)
    for n, percent in enumerate(percentage_space):
        modified_image, image_scales = preprocessor_fn(input_image, image_size)
        num_pixels_selected = int(num_pixels * percent)
        change_pixels = sorted_indices[:num_pixels_selected]
        modified_image = modified_image[0]
        for pix in change_pixels:
            modified_image[pix[0], pix[1], :] = 0
        model = get_model(model_name, modified_image, image_size)
        modified_image = modified_image[np.newaxis]
        outputs = model(modified_image)
        detection_image, detections, _ = postprocessor_fn(
            model, outputs, image_scales, raw_image, image_size,
            explain_top5_backgrounds)
        if save_modified_images:
            save_modified_image(input_image, n, saliency.shape, change_pixels)
            plt.imsave("detection_image" + str(n) + '.jpg', detection_image)
        if len(detections) == 0 and n == 0:
            raise ValueError('Detections cannot be zero here for first run')
        if len(detections) and len(det_matching_interest_det):
            all_boxes = get_evaluation_details(detections, 'corners')

            metrics = get_flipstatus_maxprob_regerror(
                all_boxes, det_matching_interest_det, percent)
            flag_label_flip, max_prob, reg_error = metrics
            if flag_label_flip:
                max_prob_list[n] = 0
                reg_error_list[n] = np.inf
                return percent, max_prob_list, reg_error_list
            else:
                max_prob_list[n] = max_prob
                reg_error_list[n] = reg_error
                continue

        else:
            max_prob_list[n] = 0
            reg_error_list[n] = np.inf
            return percent, max_prob_list, reg_error_list
    return 1, max_prob_list, reg_error_list


@gin.configurable
def coco_eval_ap50(image_index, all_boxes, result_file, percent,
                   coco_annotation_file):
    eval_json = []
    for box in all_boxes:
        eval_entry = {'image_id': image_index, 'category_id': box[4],
                      'bbox': box[:4], 'score': float(box[5])}
        eval_json.append(eval_entry)
    try:
        os.remove(result_file)
    except OSError:
        pass
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(eval_json, f, ensure_ascii=False, indent=4)
    ap_50cent = get_coco_metrics(result_file, coco_annotation_file)
    ap_50cent = round(ap_50cent, 3)
    LOGGER.info('AP 50 at modification percentage %s: %s' % (
        round(percent, 2), ap_50cent))
    return ap_50cent


def eval_object_ap_curve(
        saliency, raw_image, preprocessor_fn, postprocessor_fn, image_size=512,
        model_name='SSD512', image_index=None, ap_curve_linspace=10,
        explain_top5_backgrounds=False, result_file='ap_curve.json',
        save_modified_images=False):
    num_pixels = saliency.size
    percentage_space = np.linspace(0, 1, ap_curve_linspace)
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
        model = get_model(model_name, modified_image, image_size)
        modified_image = modified_image[np.newaxis]
        outputs = model(modified_image)
        detection_image, detections, _ = postprocessor_fn(
            model, outputs, image_scales, raw_image, image_size,
            explain_top5_backgrounds)
        if save_modified_images:
            save_modified_image(input_image, n, saliency.shape, change_pixels)
            plt.imsave("detection_image" + str(n) + '.jpg', detection_image)
        if len(detections) == 0 and n == 0:
            raise ValueError('Detections cannot be zero here for first run')
        if len(detections):
            all_boxes = get_evaluation_details(detections)
            ap_50cent = coco_eval_ap50(image_index, all_boxes, result_file,
                                       percent)
            ap_curve.append(ap_50cent)
        else:
            LOGGER.info('No detections. Mapping AP to 0.')
            ap_curve.append(0)
    LOGGER.info('AP curve: %s' % ap_curve)
    return ap_curve
