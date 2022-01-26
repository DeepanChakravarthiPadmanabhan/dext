import logging
import os
import time
import gin
import tensorflow as tf
import numpy as np
import psutil

from dext.explainer.utils import test_gpus, create_directories, get_model
from dext.factory.preprocess_factory import PreprocessorFactory
from dext.factory.postprocess_factory import PostprocessorFactory
from dext.explainer.explain_model import explain_all_objects
from dext.factory.inference_factory import InferenceFactory
from dext.explainer.utils import get_images_to_explain
from dext.explainer.utils import build_general_custom_model


LOGGER = logging.getLogger(__name__)


def get_random_model(model_name, percent_alter):
    model = get_model(model_name)
    conv = 0
    non_conv = 0
    total_weights = len(model.weights)
    selected_weights = int(total_weights * percent_alter)
    # Alter from the output layer weights
    for n, i in enumerate(model.weights[::-1][:selected_weights]):
        new_shape = model.weights[n].shape
        if 'gamma' in model.weights[n].name:
            model.weights[n].assign(
                tf.constant(1, tf.float32, shape=list(new_shape)))
            non_conv = non_conv + 1
        elif 'beta' in model.weights[n].name:
            model.weights[n].assign(
                tf.constant(0, tf.float32, shape=list(new_shape)))
            non_conv = non_conv + 1
        elif 'WSM' in model.weights[n].name:
            model.weights[n].assign(
                tf.constant(1, tf.float32, shape=list(new_shape)))
            non_conv = non_conv + 1
        elif 'mean' in model.weights[n].name:
            model.weights[n].assign(
                tf.constant(0, tf.float32, shape=list(new_shape)))
            non_conv = non_conv + 1
        elif 'variance' in model.weights[n].name:
            model.weights[n].assign(
                tf.constant(1, tf.float32, shape=list(new_shape)))
            non_conv = non_conv + 1
        elif 'bias' in model.weights[n].name:
            model.weights[n].assign(
                tf.constant(0, tf.float32, shape=list(new_shape)))
            non_conv = non_conv + 1
        else:
            model.weights[n].assign(tf.Variable(
                tf.keras.initializers.GlorotUniform()(shape=list(new_shape),
                                                      dtype=tf.float32)))
            conv = conv + 1
    LOGGER.info('No. of conv, non-conv and total weights: %s, %s, %s' %
                (conv, non_conv, total_weights))
    return model


@gin.configurable
def randomization_test(
        model_name, explain_mode, dataset_name, data_split, data_split_name,
        raw_image_path, image_size, class_layer_name, reg_layer_name,
        to_explain, interpretation_method, visualize_object_index,
        visualize_box_offset, cascade_study, randomize_weights_percent,
        random_linspace, num_images, save_saliency_images,
        save_explanation_images, continuous_run, explain_top5_backgrounds,
        load_type, result_dir):
    start_time = time.time()
    process = psutil.Process(os.getpid())
    test_gpus()
    preprocessor_fn = PreprocessorFactory(model_name).factory()
    postprocessor_fn = PostprocessorFactory(model_name).factory()
    inference_fn = InferenceFactory(model_name).factory()
    pure_model = get_model(model_name)
    if model_name != 'FasterRCNN':
        prior_boxes = pure_model.prior_boxes
    else:
        prior_boxes = None
    to_be_explained = None
    if cascade_study:
        randomize_weights_percent = np.linspace(0, 1, random_linspace)
    else:
        randomize_weights_percent = [randomize_weights_percent]
    for percent_alter in randomize_weights_percent:
        random_model = get_random_model(model_name, percent_alter)
        custom_model = build_general_custom_model(
            random_model, class_layer_name, reg_layer_name)
        out_result_dir = os.path.join(
            result_dir,
            model_name + '_' + interpretation_method + '_random' +
            str(int(percent_alter * 100)))
        create_directories(out_result_dir, False, save_saliency_images,
                           save_explanation_images)
        if not to_be_explained:
            to_be_explained = get_images_to_explain(
                explain_mode,  dataset_name, data_split, data_split_name,
                raw_image_path, num_images, continuous_run, out_result_dir)
        for count, data in enumerate(to_be_explained):
            raw_image_path = data["image"]
            image_index = data["image_index"]
            LOGGER.info('%%% BEGIN EXPLANATION MODULE %%%')
            LOGGER.info('Explaining image count: %s' % str(count + 1))
            LOGGER.info("Explanation input image ID: %s" % str(image_index))
            # forward pass - get model outputs for input image
            forward_pass_outs = inference_fn(
                pure_model, raw_image_path, preprocessor_fn, postprocessor_fn,
                image_size, explain_top5_backgrounds)
            detection_image = forward_pass_outs[0]
            detections = forward_pass_outs[1]
            box_index = forward_pass_outs[2]
            LOGGER.info("Detections: %s" % detections)
            if len(detections):
                if visualize_object_index == 'all':
                    objects_to_analyze = list(range(1, len(detections) + 1))
                else:
                    objects_to_analyze = [int(visualize_object_index)]
                explain_all_objects(
                    objects_to_analyze, raw_image_path, image_size,
                    preprocessor_fn, detections, detection_image,
                    interpretation_method, box_index, to_explain,
                    out_result_dir, class_layer_name, reg_layer_name,
                    visualize_box_offset, model_name, image_index,
                    save_saliency_images, save_explanation_images,
                    custom_model, prior_boxes, dataset_name, load_type)
            else:
                LOGGER.info("No detections to analyze.")
        end_time = time.time()
        memory_profile_in_mb = process.memory_info().rss / 1024 ** 2
        LOGGER.info('Memory profiler: %s' % memory_profile_in_mb)
        LOGGER.info('Time taken: %s' % (end_time - start_time))
        LOGGER.info('%%% INTERPRETATION DONE %%%')
    LOGGER.info('%%%%% RANDOMIZATION TEST DONE %%%%%')
