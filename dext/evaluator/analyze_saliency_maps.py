import os
import logging
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import gin

from dext.explainer.utils import resize_box
from dext.explainer.utils import get_saliency_mask
from dext.evaluator.utils import get_evaluation_details
from dext.evaluator.coco_evaluation import get_coco_metrics
from paz.backend.boxes import compute_iou
from dext.evaluator.utils import get_category_id
from dext.utils.get_image import get_image
from dext.factory.preprocess_factory import PreprocessorFactory
from dext.factory.postprocess_factory import PostprocessorFactory
from dext.interpretation_method.convert_box.utils import (
    convert_to_image_coordinates)
from paz.backend.boxes import to_center_form

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
    """
    Returns centroid in mathematical coordinates
    and not in computer coordinates.
    """
    try:
        M = cv2.moments(mask_2d)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return cx, cy
    except ZeroDivisionError:
        return 0, 0


def calculate_variance(mask_2d):
    return np.var(mask_2d)


def analyze_saliency_maps(detection, raw_image_path, saliency_map):
    image = get_image(raw_image_path)
    box = resize_box(detection, image.shape, saliency_map.shape)
    mask_2d = get_saliency_mask(saliency_map)
    iou = calculate_saliency_iou(mask_2d, box)
    centroid = calculate_centroid(mask_2d)
    variance = calculate_variance(mask_2d)
    return iou, centroid, variance


def get_regerror(interest_det, interest_gt):
    det_box = interest_det[:4]
    gt_box = interest_gt[:4]
    min_distance = np.sqrt((det_box[0] - gt_box[0]) ** 2 +
                           (det_box[1] - gt_box[1]) ** 2)
    max_distance = np.sqrt((det_box[2] - gt_box[2]) ** 2 +
                           (det_box[3] - gt_box[3]) ** 2)
    total_distance_error = min_distance + max_distance
    return total_distance_error


def get_xywh_error(interest_det, interest_gt):
    interest_gt = np.array(interest_gt[:4])[np.newaxis]
    interest_det = interest_det[np.newaxis]
    interest_gt_center = to_center_form(interest_gt)
    interest_det_center = to_center_form(interest_det)
    x_error = abs(interest_det_center[0][0] - interest_gt_center[0][0])
    y_error = abs(interest_det_center[0][1] - interest_gt_center[0][1])
    w_error = abs(interest_det_center[0][2] - interest_gt_center[0][2])
    h_error = abs(interest_det_center[0][3] - interest_gt_center[0][3])
    return x_error, y_error, w_error, h_error


def get_flipstatus_maxprob_regerror(all_det_boxes, interest_gt):
    iou = compute_iou(interest_gt[:4], np.array(all_det_boxes)[:, :4])
    max_arg = np.argmax(iou)
    interest_gt_class = int(interest_gt[4])
    interest_det = all_det_boxes[max_arg]
    interest_det_class = int(interest_det[4])
    class_match = interest_gt_class == interest_det_class
    if class_match:
        max_prob = all_det_boxes[max_arg][-1]
        reg_error_pixels = get_regerror(interest_det, interest_gt)
        reg_error_iou = iou[max_arg]
        x_error, y_error, w_error, h_error = get_xywh_error(
            np.array(interest_det), np.array(interest_gt))
        LOGGER.debug('No flip, IoU: %s, Maxprob: %s, Regerror: %s, '
                     'GT: %s, DET: %s' % (iou, max_prob, reg_error_pixels,
                                          interest_gt, interest_det))
        return (False, max_prob, reg_error_pixels, reg_error_iou,
                x_error, y_error, w_error, h_error)
    else:
        max_prob = 0
        reg_error_pixels = np.inf
        reg_error_iou = 0
        x_error, y_error, w_error, h_error = np.inf, np.inf, np.inf, np.inf
        LOGGER.debug('Flip, IoU: %s, Maxprob: %s, Regerror: %s, '
                     'GT: %s, DET: %s' % (iou, max_prob, reg_error_pixels,
                                          interest_gt, interest_det))
        return (True, max_prob, reg_error_pixels, reg_error_iou,
                x_error, y_error, w_error, h_error)


def get_interest_gt(interest_det, gt_boxes):
    iou = compute_iou(interest_det.coordinates[:4], np.array(gt_boxes)[:, :4])
    max_arg = np.argmax(iou)
    det_class_id = get_category_id(interest_det.class_name)
    if int(gt_boxes[max_arg][4] == int(det_class_id)) and iou[max_arg] > 0.4:
        return int(max_arg)
    else:
        return None


def get_all_bounded_errors(outputs, interest_gt, model_name, image_size,
                           image_scale, box_index, prior_boxes, object_index,
                           original_shape):
    outputs = outputs[0]
    selection = (0, int(box_index[object_index][0]))
    box = convert_to_image_coordinates(model_name, outputs, prior_boxes,
                                       selection, image_size, image_scale,
                                       original_shape, to_ic=True)
    reg_error_in_pixels = str(get_regerror(box, interest_gt))
    reg_error_in_iou = str(compute_iou(interest_gt,
                                       box[:4].numpy()[np.newaxis])[0])
    class_confidence = str(outputs[
        int(box_index[object_index][0]),
        int(box_index[object_index][1]) + 4].numpy())
    x, y, w, h = get_xywh_error(box[:4].numpy(), interest_gt)
    return class_confidence, reg_error_in_pixels, reg_error_in_iou, x, y, w, h


def get_interest_det(detection):
    det_interest_box = list(detection.coordinates[:4])
    det_interest_score = detection.score
    det_interest_class = int(get_category_id(detection.class_name))
    det_interest = det_interest_box
    det_interest += [det_interest_class, det_interest_score]
    return det_interest


def eval_bounded_errors(saliency, raw_image_path, detection, image_size=512,
                        model_name='EFFICIENTDETD0', ap_curve_linspace=10,
                        explain_top5_backgrounds=False,
                        save_modified_images=True,
                        image_adulteration_method='subzero', result_dir=None,
                        model=None, object_index=None):
    if model_name != 'FasterRCNN':
        prior_boxes = model.prior_boxes
    else:
        prior_boxes = None

    preprocessor_fn = PreprocessorFactory(model_name).factory()
    postprocessor_fn = PostprocessorFactory(model_name).factory()
    det_matching_interest_det = get_interest_det(detection)
    LOGGER.info('Evaluating BOUNDED ERRORS on detection: %s'
                % det_matching_interest_det)

    num_pixels = saliency.size
    percentage_space = np.linspace(0, 1, ap_curve_linspace)
    sorted_saliency = (-saliency).argsort(axis=None, kind='mergesort')
    sorted_flat_indices = np.unravel_index(sorted_saliency, saliency.shape)
    sorted_indices = np.vstack(sorted_flat_indices).T

    class_conf = []
    complete_reg_error_pixels = []
    complete_reg_error_iou = []
    x_error_pixels = []
    y_error_pixels = []
    w_error_pixels = []
    h_error_pixels = []
    visualizer_index = None

    raw_image_modifier = get_image(raw_image_path)
    original_shape = raw_image_modifier.shape
    for n, percent in enumerate(percentage_space[:-1]):
        resized_image, image_scales = preprocessor_fn(
            raw_image_modifier, image_size, True)
        num_pixels_selected = int(num_pixels * percent)
        change_pixels = sorted_indices[:num_pixels_selected]
        resized_image = resized_image[0].astype('uint8')

        if image_adulteration_method == 'inpainting':
            mask = np.zeros(saliency.shape).astype('uint8')
            mask[change_pixels[:, 0], change_pixels[:, 1]] = 1
            modified_image = cv2.inpaint(resized_image, mask, 3,
                                         cv2.INPAINT_TELEA)
        else:
            resized_image[change_pixels[:, 0], change_pixels[:, 1], :] = 0
            modified_image = resized_image

        input_image, _ = preprocessor_fn(modified_image, image_size)
        outputs = model(input_image)
        detection_image, detections, box_index = postprocessor_fn(
            model, outputs, image_scales, get_image(raw_image_path),
            image_size, explain_top5_backgrounds)
        if percent == 0:
            visualizer_index = box_index
        if save_modified_images:
            modified_images_dir = os.path.join(result_dir, "modified_images")
            plt.imsave(modified_images_dir +
                       "/modified_image" + str(n) + ".jpg", modified_image)
            plt.imsave(modified_images_dir +
                       "/detection_flip" + str(n) + '.jpg', detection_image)
        del detection_image
        del modified_image

        errors = get_all_bounded_errors(
            outputs, det_matching_interest_det, model_name, image_size,
            image_scales, visualizer_index, prior_boxes, object_index,
            original_shape)
        class_conf.append(errors[0])
        complete_reg_error_pixels.append(errors[1])
        complete_reg_error_iou.append(errors[2])
        x_error_pixels.append(errors[3])
        y_error_pixels.append(errors[4])
        w_error_pixels.append(errors[5])
        h_error_pixels.append(errors[6])

    all_errors = (class_conf, complete_reg_error_pixels,
                  complete_reg_error_iou, x_error_pixels, y_error_pixels,
                  w_error_pixels, h_error_pixels)
    return all_errors


def eval_realistic_errors(saliency, raw_image_path, detection, image_size=512,
                          model_name='EFFICIENTDETD0', ap_curve_linspace=10,
                          explain_top5_backgrounds=False,
                          save_modified_images=True,
                          image_adulteration_method='subzero', result_dir=None,
                          model=None):
    preprocessor_fn = PreprocessorFactory(model_name).factory()
    postprocessor_fn = PostprocessorFactory(model_name).factory()
    det_matching_interest_det = get_interest_det(detection)
    LOGGER.info('Evaluating REALISTIC ERRORS on detection: %s'
                % det_matching_interest_det)

    num_pixels = saliency.size
    percentage_space = np.linspace(0, 1, ap_curve_linspace)
    sorted_saliency = (-saliency).argsort(axis=None, kind='mergesort')
    sorted_flat_indices = np.unravel_index(sorted_saliency, saliency.shape)
    sorted_indices = np.vstack(sorted_flat_indices).T

    max_prob_list = [0, ] * len(percentage_space)
    reg_error_pixels_list = [np.inf, ] * len(percentage_space)
    reg_error_iou_list = [0, ] * len(percentage_space)
    x_error_pixels_list = [np.inf, ] * len(percentage_space)
    y_error_pixels_list = [np.inf, ] * len(percentage_space)
    w_error_pixels_list = [np.inf, ] * len(percentage_space)
    h_error_pixels_list = [np.inf, ] * len(percentage_space)

    raw_image_modifier = get_image(raw_image_path)
    for n, percent in enumerate(percentage_space[:-1]):
        resized_image, image_scales = preprocessor_fn(
            raw_image_modifier, image_size, True)
        num_pixels_selected = int(num_pixels * percent)
        change_pixels = sorted_indices[:num_pixels_selected]
        resized_image = resized_image[0].astype('uint8')

        if image_adulteration_method == 'inpainting':
            mask = np.zeros(saliency.shape).astype('uint8')
            mask[change_pixels[:, 0], change_pixels[:, 1]] = 1
            modified_image = cv2.inpaint(resized_image, mask, 3,
                                         cv2.INPAINT_TELEA)
        else:
            resized_image[change_pixels[:, 0], change_pixels[:, 1], :] = 0
            modified_image = resized_image

        input_image, _ = preprocessor_fn(modified_image, image_size)
        outputs = model(input_image)
        detection_image, detections, _ = postprocessor_fn(
            model, outputs, image_scales, get_image(raw_image_path),
            image_size, explain_top5_backgrounds)
        if save_modified_images:
            modified_images_dir = os.path.join(result_dir, "modified_images")
            plt.imsave(modified_images_dir +
                       "/modified_image" + str(n) + ".jpg", modified_image)
            plt.imsave(modified_images_dir +
                       "/detection_flip" + str(n) + '.jpg', detection_image)
        del detection_image
        del modified_image

        if len(detections) == 0 and n == 0:
            raise ValueError('Detections cannot be zero here for first run')

        if len(detections) and len(det_matching_interest_det):
            all_boxes = get_evaluation_details(detections, 'corners')
            metrics = get_flipstatus_maxprob_regerror(
                all_boxes, det_matching_interest_det)
            (label_flip, max_prob, reg_error_pixels, reg_error_iou,
             x_error, y_error, w_error, h_error) = metrics
            if label_flip:
                max_prob_list[n] = 0
                reg_error_pixels_list[n] = np.inf
                reg_error_iou_list[n] = 0
                x_error_pixels_list[n] = np.inf
                y_error_pixels_list[n] = np.inf
                w_error_pixels_list[n] = np.inf
                h_error_pixels_list[n] = np.inf
                return (percent, max_prob_list, reg_error_pixels_list,
                        reg_error_iou_list, x_error_pixels_list,
                        y_error_pixels_list, w_error_pixels_list,
                        h_error_pixels_list)
            else:
                max_prob_list[n] = max_prob
                reg_error_pixels_list[n] = reg_error_pixels
                reg_error_iou_list[n] = reg_error_iou
                x_error_pixels_list[n] = x_error
                y_error_pixels_list[n] = y_error
                w_error_pixels_list[n] = w_error
                h_error_pixels_list[n] = h_error
                continue
        else:
            max_prob_list[n] = 0
            reg_error_pixels_list[n] = np.inf
            reg_error_iou_list[n] = 0
            x_error_pixels_list[n] = np.inf
            y_error_pixels_list[n] = np.inf
            w_error_pixels_list[n] = np.inf
            h_error_pixels_list[n] = np.inf
            return (percent, max_prob_list, reg_error_pixels_list,
                    reg_error_iou_list, x_error_pixels_list,
                    y_error_pixels_list, w_error_pixels_list,
                    h_error_pixels_list)

    return (1, max_prob_list, reg_error_pixels_list, reg_error_iou_list,
            x_error_pixels_list, y_error_pixels_list, w_error_pixels_list,
            h_error_pixels_list)


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


def eval_object_ap_curve(saliency, raw_image_path, image_size=512,
                         model_name='SSD512', image_index=None,
                         ap_curve_linspace=10, explain_top5_backgrounds=False,
                         save_modified_images=False,
                         image_adulteration_method='inpaint',
                         result_dir=None, coco_result_file='ap_curve.json',
                         model=None):
    preprocessor_fn = PreprocessorFactory(model_name).factory()
    postprocessor_fn = PostprocessorFactory(model_name).factory()
    num_pixels = saliency.size
    percentage_space = np.linspace(0, 1, ap_curve_linspace)
    sorted_saliency = (-saliency).argsort(axis=None, kind='mergesort')
    sorted_flat_indices = np.unravel_index(sorted_saliency, saliency.shape)
    sorted_indices = np.vstack(sorted_flat_indices).T
    ap_curve = [0, ] * len(percentage_space)
    raw_image_modifier = get_image(raw_image_path)
    for n, percent in enumerate(percentage_space[:-1]):
        resized_image, image_scales = preprocessor_fn(
            raw_image_modifier, image_size, True)
        num_pixels_selected = int(num_pixels * percent)
        change_pixels = sorted_indices[:num_pixels_selected]
        resized_image = resized_image[0].astype('uint8')
        if image_adulteration_method == 'inpainting':
            mask = np.zeros(saliency.shape).astype('uint8')
            mask[change_pixels[:, 0], change_pixels[:, 1]] = 1
            modified_image = cv2.inpaint(resized_image, mask, 3,
                                         cv2.INPAINT_TELEA)
        else:
            resized_image[change_pixels[:, 0], change_pixels[:, 1], :] = 0
            modified_image = resized_image
        input_image, _ = preprocessor_fn(modified_image, image_size)
        outputs = model(input_image)
        detection_image, detections, _ = postprocessor_fn(
            model, outputs, image_scales, get_image(raw_image_path),
            image_size, explain_top5_backgrounds)
        if save_modified_images:
            modified_images_dir = os.path.join(result_dir, "modified_images")
            plt.imsave(modified_images_dir +
                       "/modified_image" + str(n) + ".jpg", modified_image)
            plt.imsave(modified_images_dir +
                       "/detection_flip" + str(n) + '.jpg', detection_image)
        del detection_image
        del modified_image
        if len(detections) == 0 and n == 0:
            raise ValueError('Detections cannot be zero here for first run')
        if len(detections):
            all_boxes = get_evaluation_details(detections)
            ap_50cent = coco_eval_ap50(
                image_index, all_boxes, coco_result_file, percent)
            ap_curve[n] = ap_50cent
        else:
            LOGGER.info('No detections. Mapping AP to 0.')
    LOGGER.info('AP curve: %s' % ap_curve)
    return ap_curve
