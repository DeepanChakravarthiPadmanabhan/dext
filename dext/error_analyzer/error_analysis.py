import os
import logging

import numpy as np
import psutil
import time
import matplotlib.pyplot as plt

from dext.explainer.utils import create_directories, get_model
from dext.factory.preprocess_factory import PreprocessorFactory
from dext.factory.postprocess_factory import PostprocessorFactory
from dext.factory.inference_factory import InferenceFactory
from dext.postprocessing.detection_visualization import plot_gt_on_detection
from dext.explainer.utils import get_images_to_explain
from dext.explainer.utils import build_general_custom_model
from dext.factory.interpretation_method_factory import ExplainerFactory
from dext.error_analyzer.utils import get_scaled_gt, get_grad_times_input
from dext.error_analyzer.utils import get_detection_list, get_tp_gt_det
from dext.error_analyzer.utils import get_closest_outbox_to_fn
from dext.error_analyzer.utils import get_missed_gt_det, get_poor_localization
from dext.utils.get_image import get_image
from dext.error_analyzer.utils import get_interest_neuron
from dext.utils.class_names import get_classes
from dext.postprocessing.saliency_visualization import plot_error_analyzer

LOGGER = logging.getLogger(__name__)


def generate_saliency(interpretation_method, custom_model, model_name,
                      raw_image_path, layer_name, box_features,
                      preprocessor_fn, image_size, prior_boxes, to_explain,
                      normalize_saliency, grad_times_input, image,
                      saliency_threshold, confidence, class_name,
                      image_index, box_offset, detections, object_index,
                      gts, dataset_name, error_type):
    interpretation_method_fn = ExplainerFactory(
        interpretation_method).factory()
    saliency, saliency_stats = interpretation_method_fn(
        custom_model, model_name, raw_image_path,
        interpretation_method, layer_name, box_features,
        preprocessor_fn, image_size, prior_boxes=prior_boxes,
        explaining=to_explain, normalize=normalize_saliency)

    if grad_times_input:
        saliency = get_grad_times_input(saliency, image)

    if saliency_threshold:
        saliency[saliency <= saliency_threshold] = 0
    LOGGER.info('Saliency stats: %s, %s', saliency.shape, saliency_stats)
    fig = plot_error_analyzer(
        raw_image_path, saliency, confidence, class_name, to_explain,
        interpretation_method, model_name, saliency_stats, box_offset,
        detections, object_index, gts, error_type, dataset_name)
    fig.savefig('sal_' + str(image_index) + '.jpg')


def analyze_errors(model_name, explain_mode, dataset_name, data_split,
                   data_split_name, raw_image_path, image_size=512,
                   class_layer_name=None, reg_layer_name=None,
                   to_explain="Classification",
                   interpretation_method="IntegratedGradients",
                   visualize_object_index=None, visualize_box_offset=None,
                   visualize_class='dog', num_images=1,
                   save_saliency_images=True, save_explanation_images=True,
                   continuous_run=False, plot_gt=False,
                   analyze_error_type='missed', use_own_class=False,
                   saliency_threshold=None, grad_times_input=False,
                   missed_with_gt=False, result_dir='images/error_analysis/',
                   save_modified_images=False):
    start_time = time.time()
    process = psutil.Process(os.getpid())

    result_dir = os.path.join(result_dir,
                              model_name + '_' + interpretation_method)
    create_directories(result_dir, save_modified_images, save_saliency_images,
                       save_explanation_images)

    preprocessor_fn = PreprocessorFactory(model_name).factory()
    postprocessor_fn = PostprocessorFactory(model_name).factory()
    inference_fn = InferenceFactory(model_name).factory()

    model = get_model(model_name)
    custom_model = build_general_custom_model(model, class_layer_name,
                                              reg_layer_name)

    class_names = get_classes('COCO', model_name)

    if model_name != 'FasterRCNN':
        prior_boxes = model.prior_boxes
    else:
        prior_boxes = None

    if to_explain == 'Classification':
        layer_name = class_layer_name
    else:
        layer_name = reg_layer_name

    normalize_saliency = not grad_times_input

    if isinstance(visualize_class, str) and use_own_class:
        visualize_class = class_names.index(visualize_class)

    to_be_explained = get_images_to_explain(
        explain_mode,  dataset_name, data_split, data_split_name,
        raw_image_path, num_images, continuous_run, result_dir)
    data = to_be_explained[0]
    raw_image_path = data["image"]
    image = get_image(raw_image_path)
    image_index = data["image_index"]
    gt = data['boxes']
    gt_list = get_scaled_gt(gt, data['width'], data['height'], model_name)
    LOGGER.info('GT loaded: %s' % gt_list)
    LOGGER.info('%%% BEGIN EXPLANATION MODULE %%%')
    LOGGER.info("Explanation input image ID: %s" % str(image_index))

    # forward pass - get model outputs for input image
    forward_pass_outs = inference_fn(
        model, raw_image_path, preprocessor_fn, postprocessor_fn,
        image_size)
    detection_image = forward_pass_outs[0]
    detections = forward_pass_outs[1]
    box_index = forward_pass_outs[2]
    outputs = forward_pass_outs[3]
    LOGGER.info("Detections: %s" % detections)
    LOGGER.info("Box index of detections: %s" % box_index)

    if len(detections):
        if plot_gt and dataset_name == 'VOC':
            detection_image = plot_gt_on_detection(detection_image, gt)
            LOGGER.info('No. of. detections: %s, No. of. GT labels: %s' %
                        (len(detections), len(gt_list)))
            plt.imsave('det_' + str(image_index) + '.jpg',
                       detection_image)

        # metrics for analysis
        det_list = get_detection_list(detections, model_name)

        # Det with correct pre (TP) and gt matching correct det (GT_TP)
        tp_list, gt_tp_list, tp_iou = get_tp_gt_det(
            det_list, gt_list)
        poor_localization_tp = get_poor_localization(tp_iou)

        # Det with wrong pred (FP) and missed GT (FN)
        fp_list, fn_list = get_missed_gt_det(tp_list, gt_tp_list,
                                             det_list, gt_list)
        LOGGER.info('TP detection index: %s' % tp_list)
        LOGGER.info('FP detection index: %s' % fp_list)
        LOGGER.info('FN detection index: %s' % fn_list)
        LOGGER.info('GT as TP, GT index: %s' % gt_tp_list)
        LOGGER.info('GT as detection index, GT index: %s' % gt_tp_list)
        LOGGER.info('TP IoUs: %s' % tp_iou)
        LOGGER.info('TP poor localization: %s' % poor_localization_tp)

        if analyze_error_type == 'missed' and len(fn_list) > 0 and (
                len(fn_list) - 1 >= visualize_object_index):
            # Visually check if the detection and gt matches
            # If missed detections -- FN
            # method 1: get prior box matching the missed gt position and
            # propagate class and offset. Answers why did the best box miss?
            fn_box_index_pred, fn_box_index_gt = get_closest_outbox_to_fn(
                prior_boxes, fn_list, gt_list, model_name, detection_image,
                outputs, image_size, raw_image_path)

            if missed_with_gt:
                box_index = fn_box_index_gt
            else:
                box_index = fn_box_index_pred
            # box_index contains (0, box, classid)

            # get box index from the object index to visualize
            box_features = box_index[visualize_object_index].copy()
            box_features[-1] = get_interest_neuron(
                to_explain, box_features[-1], visualize_box_offset,
                visualize_class, use_own_class)
            LOGGER.info('Box feature for missed study: %s' % box_features)

            if to_explain == 'Classification':
                confidence = round(outputs[0, box_features[1],
                                           box_features[2]].numpy(), 3)
                class_name = class_names[box_features[2] - 4]
            else:
                idx = get_interest_neuron(
                    'Classification', box_index[visualize_object_index][-1],
                    None, visualize_class, use_own_class)
                confidence = round(
                    outputs[0, box_features[1], idx].numpy(), 3)
                class_name = class_names[idx - 4]

            generate_saliency(interpretation_method, custom_model, model_name,
                              raw_image_path, layer_name, box_features,
                              preprocessor_fn, image_size, prior_boxes,
                              to_explain, normalize_saliency, grad_times_input,
                              image, saliency_threshold, confidence,
                              class_name, image_index, visualize_box_offset,
                              detections, visualize_object_index, gt_list,
                              dataset_name, analyze_error_type)

        elif analyze_error_type == 'wrong_class' and len(fp_list) > 0 and (
                len(fp_list) - 1 >= visualize_object_index):
            # If wrong class -- FP
            # method 1: get the output box and propagate the wrong class and
            # correct class region
            # --> Finding the features resemble the wrong class feature
            selected_fp = fp_list[visualize_object_index]
            box_features = [0, int(box_index[selected_fp][0]),
                            int(box_index[selected_fp][1])]
            box_features[-1] = get_interest_neuron(
                to_explain, box_features[-1], visualize_box_offset,
                visualize_class, use_own_class)
            LOGGER.info('Box feature for wrong class study: %s' % box_features)

            if to_explain == 'Classification':
                confidence = round(outputs[0, box_features[1],
                                           box_features[2]].numpy(), 3)
                class_name = class_names[box_features[2] - 4]
            else:
                confidence = box_index[selected_fp][2]
                class_name = class_names[box_index[selected_fp][1]]

            generate_saliency(interpretation_method, custom_model, model_name,
                              raw_image_path, layer_name, box_features,
                              preprocessor_fn, image_size, prior_boxes,
                              to_explain, normalize_saliency, grad_times_input,
                              image, saliency_threshold, confidence,
                              class_name, image_index, visualize_box_offset,
                              detections, fp_list[visualize_object_index],
                              gt_list, dataset_name, analyze_error_type)

        elif analyze_error_type == 'poor_localization' and (
                len(poor_localization_tp) > 0) and (
                len(poor_localization_tp) - 1 >= visualize_object_index):
            # If poor localization -- TP but less IOU
            # method 1: get the output box and propagate the offsets
            # method 2: get the prior box matching the gt box and
            # propagate the offsets
            selected_tp = poor_localization_tp[visualize_object_index]
            box_features = [0, int(box_index[selected_tp][0]),
                            int(box_index[selected_tp][1])]
            box_features[-1] = get_interest_neuron(
                to_explain, box_features[-1], visualize_box_offset,
                visualize_class, use_own_class)
            LOGGER.info('Box feature for poor ROI study: %s' % box_features)

            if to_explain == 'Classification':
                confidence = round(outputs[0, box_features[1],
                                           box_features[2]].numpy(), 3)
                class_name = class_names[box_features[2] - 4]
            else:
                confidence = box_index[selected_tp][2]
                class_name = class_names[box_index[selected_tp][1]]

            generate_saliency(interpretation_method, custom_model, model_name,
                              raw_image_path, layer_name, box_features,
                              preprocessor_fn, image_size, prior_boxes,
                              to_explain, normalize_saliency, grad_times_input,
                              image, saliency_threshold, confidence,
                              class_name, image_index, visualize_box_offset,
                              detections,
                              poor_localization_tp[visualize_object_index],
                              gt_list, dataset_name, analyze_error_type)

        else:
            print('Analysis type is not possible')

    else:
        LOGGER.info("No detections to analyze.")
    end_time = time.time()
    memory_profile_in_mb = process.memory_info().rss / 1024 ** 2
    LOGGER.info('Memory profiler: %s' % memory_profile_in_mb)
    LOGGER.info('Time taken: %s' % (end_time - start_time))
    LOGGER.info('%%% ERROR ANALYSIS DONE %%%')
