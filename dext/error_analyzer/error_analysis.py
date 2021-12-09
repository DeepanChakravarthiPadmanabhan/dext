import os
import logging
import psutil
import time

from dext.explainer.utils import create_directories, get_model
from dext.factory.preprocess_factory import PreprocessorFactory
from dext.factory.postprocess_factory import PostprocessorFactory
from dext.factory.inference_factory import InferenceFactory
from dext.postprocessing.detection_visualization import plot_gt_on_detection
from dext.explainer.utils import get_images_to_explain
from dext.explainer.utils import build_general_custom_model
from dext.explainer.explain_model import explain_all_objects
from dext.error_analyzer.utils import find_errors

LOGGER = logging.getLogger(__name__)


def analyze_errors(model_name, explain_mode, dataset_name, data_split,
                   data_split_name, raw_image_path, image_size=512,
                   class_layer_name=None, reg_layer_name=None,
                   to_explain="Classification",
                   interpretation_method="IntegratedGradients",
                   visualize_object_index=None, visualize_box_offset=None,
                   num_images=1, save_saliency_images=True,
                   save_explanation_images=True, continuous_run=False,
                   plot_gt=False, result_dir='images/error_analysis/',
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
    to_be_explained = get_images_to_explain(
        explain_mode,  dataset_name, data_split, data_split_name,
        raw_image_path, num_images, continuous_run, result_dir)
    model = get_model(model_name)
    custom_model = build_general_custom_model(
        model, class_layer_name, reg_layer_name)
    if model_name != 'FasterRCNN':
        prior_boxes = model.prior_boxes
    else:
        prior_boxes = None
    for count, data in enumerate(to_be_explained):
        raw_image_path = data["image"]
        image_index = data["image_index"]
        gt = data['boxes']
        LOGGER.info('%%% BEGIN EXPLANATION MODULE %%%')
        LOGGER.info('Explaining image count: %s' % str(count + 1))
        LOGGER.info("Explanation input image ID: %s" % str(image_index))
        # forward pass - get model outputs for input image
        forward_pass_outs = inference_fn(
            model, raw_image_path, preprocessor_fn, postprocessor_fn,
            image_size)
        detection_image = forward_pass_outs[0]
        detections = forward_pass_outs[1]
        box_index = forward_pass_outs[2]
        LOGGER.info("Detections: %s" % detections)
        if len(detections):
            if plot_gt and dataset_name == 'VOC':
                detection_image = plot_gt_on_detection(detection_image, gt)
                LOGGER.info('No. of. detections: %s, No. of. GT labels: %s' %
                            (len(detections), len(gt)))
            if visualize_object_index == 'all':
                objects_to_analyze = list(range(1, len(detections) + 1))
            else:
                objects_to_analyze = [int(visualize_object_index)]
            explain_all_objects(
                objects_to_analyze, raw_image_path, image_size,
                preprocessor_fn, detections, detection_image,
                interpretation_method, box_index, to_explain, result_dir,
                class_layer_name, reg_layer_name, visualize_box_offset,
                model_name, image_index, save_saliency_images,
                save_explanation_images, custom_model, prior_boxes)
        else:
            LOGGER.info("No detections to analyze.")
    end_time = time.time()
    memory_profile_in_mb = process.memory_info().rss / 1024 ** 2
    LOGGER.info('Memory profiler: %s' % memory_profile_in_mb)
    LOGGER.info('Time taken: %s' % (end_time - start_time))
    LOGGER.info('%%% INTERPRETATION DONE %%%')
    print('\n')
    LOGGER.info('%%% ERROR ANALYSIS STARTS %%%')
    errors = find_errors(gt, detections, dataset_name, model_name)
    missed_detections = []
    poor_localizations = []
    wrong_classifications = []
    excess_detections = []



