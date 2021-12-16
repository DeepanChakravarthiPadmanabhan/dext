import logging
import cv2
import numpy as np

from dext.utils.class_names import get_classes, coco_to_voc
from dext.model.utils import find_image_scale
from dext.interpretation_method.convert_box.fasterrcnn_conversion import (
    get_fasterrcnn_boxes)
from dext.interpretation_method.convert_box.ssd_conversion import get_ssd_boxes
from dext.interpretation_method.convert_box.efficientdet_conversion import (
    get_efficientdet_boxes)
from dext.model.faster_rcnn.faster_rcnn_preprocess import (
    faster_rcnn_preprocess)
from dext.utils.get_image import get_image
from dext.utils.class_names import voc_to_coco
from paz.backend.boxes import compute_iou
from dext.postprocessing.saliency_visualization import (
    visualize_saliency_grayscale)

LOGGER = logging.getLogger(__name__)


def get_scaled_gt(gt, width, height, model_name):
    scaled_gt_list = []
    for normalized_gt in gt:
        scaled_gt = [normalized_gt[0] * width, normalized_gt[1] * height,
                     normalized_gt[2] * width, normalized_gt[3] * height,
                     int(normalized_gt[-1])]
        scaled_gt_list.append(scaled_gt)
    return scaled_gt_list


def get_detection_list(detections, model_name):
    class_names = get_classes('VOC', model_name)
    det_list = []
    for det in detections:
        coordinates = list(det.coordinates)
        class_id = [class_names.index(coco_to_voc[det.class_name])
                    if det.class_name in coco_to_voc else
                    class_names.index(det.class_name), ]
        det_list.append(coordinates + class_id)
    return det_list


def get_tp_gt_det(det_list, gt_list):
    det_tp_idx = []  # holds detection tp indices
    gt_tp_idx = []  # holds gt indices matching tp
    det_tp_iou = []  # holds iou of det tp
    for i in range(len(det_list)):
        iou = compute_iou(np.array(det_list)[i][:4], np.array(gt_list)[:, :4])
        idx = np.argmax(iou)
        if idx not in gt_tp_idx:
            if det_list[i][-1] == gt_list[idx][-1]:
                det_tp_idx.append(i)
                gt_tp_idx.append(idx)
                det_tp_iou.append(iou[idx])
        else:
            avail_idx = gt_tp_idx.index(idx)
            if det_tp_iou[avail_idx] < iou[idx]:
                if det_list[i][-1] == gt_list[idx][-1]:
                    det_tp_iou[avail_idx] = iou[idx]
                    gt_tp_idx[avail_idx] = idx
                    det_tp_idx[avail_idx] = i
    return det_tp_idx, gt_tp_idx, det_tp_iou


def get_missed_gt_det(det_tp_idx, gt_tp_idx, det_list, gt_list):
    fp_list = []
    fn_list = []
    for i in range(len(det_list)):
        if i not in det_tp_idx:
            fp_list.append(i)
    for i in range(len(gt_list)):
        if i not in gt_tp_idx:
            fn_list.append(i)
    return fp_list, fn_list


def get_image_scale(detection_image, image_size):
    new_image_shape = (1, image_size, image_size, 3)
    original_image_shape = detection_image.shape
    image_scale = find_image_scale(original_image_shape, new_image_shape)
    return image_scale


def get_output_boxes(outputs, prior_boxes, image_size, detection_image,
                     model_name):
    outputs = outputs[0]
    if model_name == 'SSD512':
        image_scale = get_image_scale(detection_image, image_size)
        out_boxes = get_ssd_boxes(outputs, prior_boxes, image_size,
                                  image_scale, True)
    elif 'EFFICIENTDET' in model_name:
        image_scale = get_image_scale(detection_image, image_size)
        out_boxes = get_efficientdet_boxes(outputs, prior_boxes, image_size,
                                           image_scale, True)
    elif model_name == 'FasterRCNN':
        original_image_shape = detection_image.shape
        _, image_scale = faster_rcnn_preprocess(detection_image, image_size,
                                                only_resize=True)
        out_boxes = get_fasterrcnn_boxes(outputs, original_image_shape,
                                         image_size, image_scale, True)
    else:
        raise ValueError('Model name not in prior box processing.')
    return out_boxes


def get_closest_outbox_to_fn(prior_boxes, fn_list, gt_list, model_name,
                             detection_image, outputs, image_size,
                             raw_image_path):
    image = get_image(raw_image_path)
    out_boxes = get_output_boxes(outputs, prior_boxes, image_size,
                                 detection_image, model_name)
    class_names_voc = get_classes('VOC', model_name)
    class_names_coco = get_classes('COCO', model_name)

    fn_box_index_pred = []
    fn_box_index_gt = []
    for i in range(len(fn_list)):
        # find output box closest with gt box
        iou = compute_iou(gt_list[fn_list[i]][:4], out_boxes)
        idx = np.argmax(iou)
        print("IoU of box closest: ", iou[idx], idx)
        # output box closest with gt box
        closest_output_box = out_boxes[idx, :4]
        gt_box = gt_list[fn_list[i]][:4]
        print('Closest output box and gt box: ',
              closest_output_box.numpy(), gt_box)

        class_id_pred = np.argmax(outputs[0][idx][4:])
        fn_box_index_pred.append([0, idx, class_id_pred])

        class_name_gt = class_names_voc[gt_list[fn_list[i]][-1]]
        if class_name_gt in voc_to_coco:
            class_name_gt = voc_to_coco[class_name_gt]
        class_id_gt = class_names_coco.index(class_name_gt)
        fn_box_index_gt.append([0, idx, class_id_gt])

        print('Closest output box and gt box class: ',
              class_id_pred, class_id_gt)

        # draw gt (red) and closest box (blue) on image
        image = cv2.rectangle(
            image, (int(gt_box[0]), int(gt_box[1])),
            (int(gt_box[2]), int(gt_box[3])), (255, 0, 0), 1, cv2.LINE_AA)
        image = cv2.rectangle(
            image, (int(closest_output_box[0]), int(closest_output_box[1])),
            (int(closest_output_box[2]), int(closest_output_box[3])),
            (0, 0, 255), 1, cv2.LINE_AA)
    # plt.imsave('gt_drawn.jpg', image)

    return fn_box_index_pred, fn_box_index_gt


def get_interest_neuron(explaining, neuron, visualize_box_offset,
                        visualize_class, use_own_class=False):
    if explaining == 'Classification':
        if use_own_class:
            return visualize_class + 4
        else:
            return neuron + 4
    else:
        box_arg_to_index = {'x_min': 0, 'y_min': 1, 'x_max': 2, 'y_max': 3}
        return box_arg_to_index[visualize_box_offset]


def get_poor_localization(tp_list):
    poor_localization_tp = [i for i in tp_list if i < 0.7]
    return poor_localization_tp


def get_grad_times_input(saliency, image):
    saliency = cv2.resize(saliency[0], (image.shape[1], image.shape[0]))
    saliency = np.multiply(image, saliency)
    saliency = saliency[np.newaxis]
    saliency = visualize_saliency_grayscale(saliency)
    return saliency
