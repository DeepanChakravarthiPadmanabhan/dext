import logging
import os
import time
import numpy as np
import psutil
import gc
from memory_profiler import profile

from dext.explainer.utils import test_gpus, create_directories, get_model
from dext.factory.preprocess_factory import PreprocessorFactory
from dext.factory.postprocess_factory import PostprocessorFactory
from dext.factory.interpretation_method_factory import ExplainerFactory
from dext.factory.inference_factory import InferenceFactory
from dext.postprocessing.saliency_visualization import plot_all
from dext.postprocessing.detection_visualization import plot_gt_on_detection
from dext.explainer.utils import get_box_feature_index
from dext.explainer.utils import get_explaining_info
from dext.explainer.utils import get_images_to_explain
from dext.explainer.utils import get_model_class_name
from dext.explainer.utils import write_record
from dext.explainer.utils import build_general_custom_model

LOGGER = logging.getLogger(__name__)


def get_single_saliency(
        interpretation_method, box_index, explaining, visualize_object_index,
        visualize_box_offset, model_name, raw_image_path, layer_name,
        preprocessor_fn, image_size, custom_model):
    # select - get index to visualize saliency input image
    box_features = get_box_feature_index(
        box_index, explaining, visualize_object_index, model_name,
        visualize_box_offset)
    # interpret - apply interpretation method
    interpretation_method_fn = ExplainerFactory(
        interpretation_method).factory()
    saliency = interpretation_method_fn(
        custom_model, model_name, raw_image_path, interpretation_method,
        layer_name, box_features, preprocessor_fn, image_size)
    return saliency


def save_saliency_record(
        result_dir, image_index, object_index, coordinates, score, class_name,
        explaining, box_offset, save_path, raw_image_path):
    record = [image_index, object_index, coordinates, score, class_name,
              explaining, box_offset, save_path, raw_image_path]
    write_record(record, "saliency_image_paths", result_dir)


def explain_single_object(raw_image_path, image_size, preprocessor_fn,
                          detections, interpretation_method, box_index,
                          result_dir, explaining_info, model_name, image_index,
                          class_name, class_confidence, save_saliency_images,
                          custom_model):
    saliency_list = []
    saliency_stat_list = []
    for info in zip(*explaining_info):
        object_index = info[0]
        explaining = info[1]
        layer_name = info[2]
        box_offset = info[3]
        LOGGER.info("Generating - image index: %s, explaining: %s, offset: %s"
                    % (str(image_index), explaining, box_offset))
        saliency, saliency_stat = get_single_saliency(
            interpretation_method, box_index, explaining, object_index,
            box_offset, model_name, raw_image_path, layer_name,
            preprocessor_fn, image_size, custom_model)
        if save_saliency_images:
            save_name = str(image_index) + "_" + str(object_index) + "_" + (
                explaining) + "_" + str(box_offset) + "_" + (
                interpretation_method)
            saliency_images_dir = os.path.join(result_dir, 'saliency_images')
            save_path = os.path.join(saliency_images_dir, save_name + ".npy")
            np.save(save_path, saliency)
            save_saliency_record(
                result_dir, str(image_index), str(object_index),
                detections[object_index].coordinates, class_confidence,
                class_name, explaining, box_offset, save_path, raw_image_path)
        saliency_list.append(saliency)
        saliency_stat_list.append(saliency_stat)
    LOGGER.info("Completed explaining image index: %s" % str(image_index))
    del saliency
    del saliency_stat
    gc.collect()
    return saliency_list, saliency_stat_list


def explain_all_objects(objects_to_analyze, raw_image_path, image_size,
                        preprocessor_fn, detections, detection_image,
                        interpretation_method, box_index, to_explain,
                        result_dir, class_layer_name, reg_layer_name,
                        visualize_box_offset, model_name, image_index,
                        save_saliency_images, save_explanation_images,
                        custom_model):
    for object_arg in objects_to_analyze:
        explaining_info = get_explaining_info(
            object_arg, box_index, to_explain, class_layer_name,
            reg_layer_name,
            visualize_box_offset, model_name)
        class_name = get_model_class_name(model_name, 'COCO')[
            box_index[explaining_info[0][0]][1]]
        class_confidence = box_index[explaining_info[0][0]][2]
        LOGGER.info('Explaining - image index: %s, confidence: %s, class: %s'
                    % (str(image_index), class_confidence, class_name))
        LOGGER.info("Information - %s" % (explaining_info,))
        saliency_list, saliency_stat_list = explain_single_object(
            raw_image_path, image_size, preprocessor_fn, detections,
            interpretation_method, box_index, result_dir, explaining_info,
            model_name, image_index, class_name, class_confidence,
            save_saliency_images, custom_model)
        if save_explanation_images:
            explanation_images_dir = os.path.join(
                result_dir, 'explanation_images')
            plot_all(detection_image, raw_image_path, saliency_list,
                     saliency_stat_list, class_confidence, class_name,
                     explaining_info[1], explaining_info[3], to_explain,
                     interpretation_method, model_name, "subplot",
                     explanation_images_dir, image_index, object_arg)
        del saliency_list
        del saliency_stat_list
        gc.collect()


@profile
def explain_model(model_name, explain_mode,  dataset_name, data_split,
                  data_split_name, raw_image_path, image_size=512,
                  class_layer_name=None, reg_layer_name=None,
                  to_explain="Classification", result_dir='images/results/',
                  interpretation_method="IntegratedGradients",
                  visualize_object_index=None, visualize_box_offset=None,
                  num_images=2, save_saliency_images=False,
                  save_explanation_images=False, continuous_run=False,
                  explain_top5_backgrounds=True, plot_gt=False,
                  save_modified_images=True):
    start_time = time.time()
    process = psutil.Process(os.getpid())
    test_gpus()
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
            image_size, explain_top5_backgrounds)
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
                save_explanation_images, custom_model)
        else:
            LOGGER.info("No detections to analyze.")
    end_time = time.time()
    memory_profile_in_mb = process.memory_info().rss / 1024 ** 2
    LOGGER.info('Memory profiler: %s' % memory_profile_in_mb)
    LOGGER.info('Time taken: %s' % (end_time - start_time))
    LOGGER.info('%%% INTERPRETATION DONE %%%')
