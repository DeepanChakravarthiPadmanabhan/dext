import os
import time
import psutil
import logging
import cv2
import numpy as np
import kneed
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors

from dext.explainer.utils import test_gpus, create_directories, get_model
from dext.factory.preprocess_factory import PreprocessorFactory
from dext.factory.postprocess_factory import PostprocessorFactory
from dext.factory.inference_factory import InferenceFactory
from dext.explainer.utils import get_explaining_info
from dext.utils.class_names import get_classes
from dext.explainer.utils import get_images_to_explain
from dext.explainer.explain_model import explain_single_object
from dext.explainer.utils import build_general_custom_model
from dext.explainer.utils import write_record
from dext.postprocessing.saliency_visualization import plot_all_matplotlib

LOGGER = logging.getLogger(__name__)


def explain_all_objects(objects_to_analyze, raw_image_path, image_size,
                        preprocessor_fn, detections, interpretation_method,
                        box_index, to_explain, result_dir, class_layer_name,
                        reg_layer_name, visualize_box_offset, model_name,
                        image_index, custom_model, prior_boxes,
                        k_values=[4, 13, 50, 100]):
    for object_arg in objects_to_analyze:
        explaining_info = get_explaining_info(
            object_arg, box_index, to_explain, class_layer_name,
            reg_layer_name, visualize_box_offset, model_name)
        class_name = get_classes('COCO', model_name)[
            box_index[explaining_info[0][0]][1]]
        class_confidence = box_index[explaining_info[0][0]][2]
        LOGGER.info('Explaining - image index: %s, confidence: %s, class: %s'
                    % (str(image_index), class_confidence, class_name))
        LOGGER.info("Information - %s" % (explaining_info,))
        saliency_list, saliency_stat_list = explain_single_object(
            raw_image_path, image_size, preprocessor_fn, detections,
            interpretation_method, box_index, result_dir, explaining_info,
            model_name, image_index, class_name, class_confidence,
            False, custom_model, prior_boxes)
        explanation_images_dir = os.path.join(
            result_dir, 'explanation_images')
        plot_all_matplotlib(
            detections, raw_image_path, explaining_info[0][0],
            saliency_list, saliency_stat_list, class_confidence,
            class_name, explaining_info[1], explaining_info[3],
            to_explain, interpretation_method, model_name,
            explanation_images_dir, image_index, object_arg)

        gray = np.uint8(saliency_list[0] * 255)
        # 204 because taking 0.8 <
        (thresh, blackAndWhiteImage) = cv2.threshold(gray, 204, 255,
                                                     cv2.THRESH_BINARY)
        blackAndWhiteImage = cv2.bitwise_not(blackAndWhiteImage)
        # convert black pixels to coordinates
        X = np.column_stack(np.where(blackAndWhiteImage == 0))
        eps = []
        min_points = []
        # save parameters of k knee
        for minpoints in k_values:
            neighbors = minpoints - 1
            # Nearest neighbors
            nbrs = NearestNeighbors(n_neighbors=neighbors).fit(X)
            distances, indices = nbrs.kneighbors(X)
            distance_desc = sorted(distances[:, neighbors - 1], reverse=True)
            kneedle = KneeLocator(range(1, len(distance_desc) + 1),  # x values
                                  distance_desc,  # y values
                                  S=1.0,  # parameter suggested from paper
                                  curve="convex",  # parameter from figure
                                  direction="decreasing")
            eps.append(str(kneedle.elbow))
            min_points.append(str(kneedle.knee_y))
        # write to a separate text file
        object_index = explaining_info[0][0]
        result_list = [str(image_index), str(object_index), class_name,
                       str(class_confidence),
                       detections[object_index].coordinates,
                       raw_image_path]
        result_list += eps
        result_list += min_points
        write_record(result_list,
                     model_name + '_' + interpretation_method + '.txt',
                     result_dir)


def multi_detection_visualizer(
        model_name, explain_mode, dataset_name, data_split, data_split_name,
        input_image_path, image_size, class_layer_name, reg_layer_name,
        to_explain, interpretation_method, visualize_object_index,
        visualize_box_offset, num_images, continuous_run,
        save_explanation_images=True,
        result_dir='images/results/hp_tuning'):
    start_time = time.time()
    process = psutil.Process(os.getpid())
    test_gpus()
    create_directories(result_dir, False, False, save_explanation_images)
    preprocessor_fn = PreprocessorFactory(model_name).factory()
    postprocessor_fn = PostprocessorFactory(model_name).factory()
    inference_fn = InferenceFactory(model_name).factory()
    to_be_explained = get_images_to_explain(
        explain_mode, dataset_name, data_split, data_split_name,
        input_image_path, num_images, continuous_run, result_dir)
    model = get_model(model_name)
    custom_model = build_general_custom_model(
        model, class_layer_name, reg_layer_name)
    if model_name != 'FasterRCNN':
        prior_boxes = model.prior_boxes
    else:
        prior_boxes = None
    print(to_be_explained)
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
            image_size, False)
        detection_image = forward_pass_outs[0]
        detections = forward_pass_outs[1]
        box_index = forward_pass_outs[2]
        LOGGER.info("Detections: %s" % detections)
        if len(detections):
            LOGGER.info('No. of. detections: %s, No. of. GT labels: %s' %
                        (len(detections), len(gt)))
            if visualize_object_index == 'all':
                objects_to_analyze = list(range(1, len(detections) + 1))
            else:
                objects_to_analyze = [int(visualize_object_index)]
            explain_all_objects(
                objects_to_analyze, raw_image_path, image_size,
                preprocessor_fn, detections, interpretation_method, box_index,
                to_explain, result_dir, class_layer_name, reg_layer_name,
                visualize_box_offset, model_name, image_index, custom_model,
                prior_boxes)
        else:
            LOGGER.info("No detections to analyze.")
    end_time = time.time()
    memory_profile_in_mb = process.memory_info().rss / 1024 ** 2
    LOGGER.info('Memory profiler: %s' % memory_profile_in_mb)
    LOGGER.info('Time taken: %s' % (end_time - start_time))
    LOGGER.info('%%% INTERPRETATION DONE %%%')
