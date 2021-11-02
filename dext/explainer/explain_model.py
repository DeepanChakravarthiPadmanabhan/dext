import logging
import os
import time
import json
import numpy as np
import psutil

from paz.backend.image.opencv_image import write_image

from dext.explainer.utils import get_model
from dext.factory.preprocess_factory import PreprocessorFactory
from dext.factory.postprocess_factory import PostprocessorFactory
from dext.factory.interpretation_method_factory import ExplainerFactory
from dext.factory.inference_factory import InferenceFactory
from dext.postprocessing.saliency_visualization import plot_all
from dext.explainer.utils import get_box_feature_index
from dext.explainer.utils import get_explaining_info
from dext.explainer.utils import get_images_to_explain
from dext.explainer.analyze_saliency_maps import analyze_saliency_maps
from dext.explainer.utils import get_model_class_name
from dext.explainer.postprocess_saliency import merge_saliency
from dext.explainer.analyze_saliency_maps import eval_object_ap_curve
from dext.explainer.analyze_saliency_maps import eval_numflip_maxprob_regerror


LOGGER = logging.getLogger(__name__)


def get_single_saliency(
        interpretation_method, box_index, explaining, visualize_object_index,
        visualize_box_offset, model_name, raw_image_path, layer_name,
        preprocessor_fn, image_size):
    # select - get index to visualize saliency input image
    box_features = get_box_feature_index(
        box_index, explaining, visualize_object_index, model_name,
        visualize_box_offset)
    # interpret - apply interpretation method
    interpretation_method_fn = ExplainerFactory(
        interpretation_method).factory()
    saliency = interpretation_method_fn(
        model_name, raw_image_path, interpretation_method,
        layer_name, box_features, preprocessor_fn, image_size)
    return saliency


def write_record(record, file_name, result_dir):
    file_name = os.path.join(result_dir, file_name)
    with open(file_name, 'a', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False)
        f.write(os.linesep)


def get_metrics(detections, raw_image_path, saliency, object_arg,
                preprocessor_fn, postprocessor_fn, model_name, image_size,
                explaining, ap_curve_linspace, image_index, result_dir,
                explain_top5_backgrounds, save_modified_images,
                image_adulteration_method, eval_flip, eval_ap_explain):
    if eval_flip:
        LOGGER.info('Extracting metrics for the explanation %s' % explaining)
        LOGGER.info('Performing max probability and regression error')
        (saliency_iou, saliency_centroid,
         saliency_variance) = analyze_saliency_maps(detections, raw_image_path,
                                                    saliency, object_arg)
        eval_metrics = eval_numflip_maxprob_regerror(
            saliency, raw_image_path, detections,
            preprocessor_fn, postprocessor_fn, image_size, model_name,
            object_arg, ap_curve_linspace, explain_top5_backgrounds,
            save_modified_images, image_adulteration_method)
        num_pixels_flipped, max_prob_curve, reg_error_curve = eval_metrics
        df_class_flip_entry = [str(image_index), object_arg, explaining,
                               detections[object_arg].coordinates,
                               detections[object_arg].score,
                               detections[object_arg].class_name,
                               saliency_iou,
                               saliency_centroid, saliency_variance,
                               num_pixels_flipped]
        write_record(df_class_flip_entry, 'class_flip', result_dir)
        df_max_prob_entry = [str(image_index), object_arg, num_pixels_flipped,
                             explaining, ]
        df_max_prob_entry = df_max_prob_entry + max_prob_curve
        write_record(df_max_prob_entry, 'max_prob', result_dir)
        df_reg_error_entry = [str(image_index), object_arg, num_pixels_flipped,
                              explaining, ]
        df_reg_error_entry = df_reg_error_entry + reg_error_curve
        write_record(df_reg_error_entry, 'reg_error', result_dir)

    if eval_ap_explain:
        LOGGER.info('Extracting metrics for the explanation %s' % explaining)
        LOGGER.info('Performing AP Curve evaluation')
        ap_curve = eval_object_ap_curve(
            saliency, raw_image_path, preprocessor_fn, postprocessor_fn,
            image_size, model_name, image_index, ap_curve_linspace,
            explain_top5_backgrounds, save_modified_images,
            image_adulteration_method)
        df_ap_curve_entry = [str(image_index), object_arg, explaining, ]
        df_ap_curve_entry = df_ap_curve_entry + ap_curve
        write_record(df_ap_curve_entry, 'ap_curve', result_dir)


def merge_all_maps(saliency_list, merge_method, analyze_each_maps,
                   detections, raw_image_path, preprocessor_fn,
                   postprocessor_fn, image_size, title, model_name,
                   ap_curve_linspace, image_index, result_dir,
                   explain_top5_backgrounds, save_modified_images,
                   image_adulteration_method, object_arg,
                   eval_flip, eval_ap_explain):
    combined_saliency = merge_saliency(saliency_list, merge_method)
    if analyze_each_maps:
        get_metrics(
            detections, raw_image_path, combined_saliency,
            object_arg, preprocessor_fn, postprocessor_fn, model_name,
            image_size, title, ap_curve_linspace, image_index, result_dir,
            explain_top5_backgrounds, save_modified_images,
            image_adulteration_method, eval_flip, eval_ap_explain)


def explain_single_object(
        raw_image_path, image_size, preprocessor_fn, postprocessor_fn,
        detections, detection_image, interpretation_method, object_arg,
        box_index, to_explain, result_dir, class_layer_name, reg_layer_name,
        visualize_box_offset, model_name, merge_method, image_index,
        save_explanations, analyze_each_maps, ap_curve_linspace, eval_flip,
        eval_ap_explain, merge_saliency_maps, explain_top5_backgrounds,
        save_modified_images, image_adulteration_method, evaluate_random_map):
    explaining_info = get_explaining_info(
        object_arg, box_index, to_explain, class_layer_name, reg_layer_name,
        visualize_box_offset, model_name)
    LOGGER.info('Explaining image index: %s' % str(image_index))
    LOGGER.info("Information used for explanation: %s" %
                (explaining_info,))
    object_index_list = explaining_info[0]
    explaining_list = explaining_info[1]
    layer_name_list = explaining_info[2]
    box_offset_list = explaining_info[3]
    saliency_list = []
    saliency_stat_list = []
    random_saliency_list = []
    confidence_list = []
    class_name_list = []
    explaining_info = zip(object_index_list,
                          explaining_list,
                          layer_name_list,
                          box_offset_list)
    for info in explaining_info:
        object_index = info[0]
        explaining = info[1]
        layer_name = info[2]
        box_offset = info[3]
        class_name = get_model_class_name(model_name, 'COCO')[
            box_index[object_index][1]]
        class_confidence = box_index[object_index][2]
        LOGGER.info("Generating saliency: Image index - %s, Object - %s, "
                    "Confidence - %s, Explaining - %s, Box offset - %s"
                    % (str(image_index), class_name, class_confidence,
                       explaining, box_offset))
        saliency, saliency_stat = get_single_saliency(
            interpretation_method, box_index, explaining, object_index,
            box_offset, model_name, raw_image_path,
            layer_name, preprocessor_fn, image_size)
        saliency_list.append(saliency)
        saliency_stat_list.append(saliency_stat)
        confidence_list.append(class_confidence)
        class_name_list.append(class_name)
        # analyze saliency maps
        if analyze_each_maps:
            get_metrics(
                detections, raw_image_path, saliency,
                object_index, preprocessor_fn, postprocessor_fn, model_name,
                image_size, explaining + str(box_offset), ap_curve_linspace,
                image_index, result_dir, explain_top5_backgrounds,
                save_modified_images, image_adulteration_method, eval_flip,
                eval_ap_explain)
        if evaluate_random_map:
            random_map = np.random.random((image_size, image_size))
            random_saliency_list.append(random_map)
            get_metrics(
                detections, raw_image_path, random_map,
                object_index, preprocessor_fn, postprocessor_fn, model_name,
                image_size, explaining + str(box_offset) + '_random',
                ap_curve_linspace, image_index, result_dir,
                explain_top5_backgrounds, save_modified_images,
                image_adulteration_method, eval_flip, eval_ap_explain)

    if merge_saliency_maps:
        LOGGER.info('Merging saliency maps')
        merge_all_maps(
            saliency_list, merge_method, analyze_each_maps, detections,
            raw_image_path, preprocessor_fn, postprocessor_fn,
            image_size, 'combined', model_name, ap_curve_linspace, image_index,
            result_dir, explain_top5_backgrounds, save_modified_images,
            image_adulteration_method, object_index_list[0], eval_flip,
            eval_ap_explain)
        if evaluate_random_map:
            LOGGER.info('Merging saliency maps - random baseline')
            merge_all_maps(
                random_saliency_list, merge_method, analyze_each_maps,
                detections, raw_image_path, preprocessor_fn,
                postprocessor_fn, image_size, 'random', model_name,
                ap_curve_linspace, image_index, result_dir,
                explain_top5_backgrounds, save_modified_images,
                image_adulteration_method, object_index_list[0], eval_flip,
                eval_ap_explain)

    if save_explanations:
        explanation_result_dir = os.path.join(result_dir, 'explanations')
        if not os.path.exists(explanation_result_dir):
            os.makedirs(explanation_result_dir)
        plot_all(detection_image, raw_image_path, saliency_list,
                 saliency_stat_list, confidence_list, class_name_list,
                 explaining_list, box_offset_list, to_explain,
                 interpretation_method, model_name, "subplot",
                 explanation_result_dir, image_index, object_arg)
    LOGGER.info("Box and class labels, after post-processing: %s"
                % box_index)
    LOGGER.info("Completed explaining image index: %s" % str(image_index))


def explain_all_objects(
        objects_to_analyze, raw_image_path, image_size, preprocessor_fn,
        postprocessor_fn, detections, detection_image, interpretation_method,
        box_index, to_explain, result_dir, class_layer_name, reg_layer_name,
        visualize_box_offset, model_name, merge_method, image_index,
        save_explanations, analyze_each_maps, ap_curve_linspace, eval_flip,
        eval_ap_explain, merge_saliency_maps, explain_top5_backgrounds,
        save_modified_images, image_adulteration_method, evaluate_random_map):
    for object_arg in objects_to_analyze:
        explain_single_object(
            raw_image_path, image_size, preprocessor_fn, postprocessor_fn,
            detections, detection_image, interpretation_method,
            object_arg, box_index, to_explain, result_dir, class_layer_name,
            reg_layer_name, visualize_box_offset, model_name, merge_method,
            image_index, save_explanations, analyze_each_maps,
            ap_curve_linspace, eval_flip, eval_ap_explain, merge_saliency_maps,
            explain_top5_backgrounds, save_modified_images,
            image_adulteration_method, evaluate_random_map)


def explain_model(model_name, explain_mode, raw_image_path, image_size=512,
                  class_layer_name=None, reg_layer_name=None,
                  to_explain="Classification", result_dir='images/results/',
                  interpretation_method="IntegratedGradients",
                  visualize_object_index=None, visualize_box_offset=None,
                  num_images=2, merge_method='add', save_detections=False,
                  save_explanations=False, analyze_each_maps=False,
                  ap_curve_linspace=20, eval_flip=False, eval_ap_explain=False,
                  merge_saliency_maps=False, explain_top5_backgrounds=True,
                  save_modified_images=True, image_adulteration_method=None,
                  evaluate_random_map=True):
    start_time = time.time()
    process = psutil.Process(os.getpid())
    result_dir = os.path.join(result_dir,
                              model_name + '_' + interpretation_method)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if save_modified_images:
        if not os.path.exists(os.path.join(result_dir, 'modified_images')):
            os.makedirs(os.path.join(result_dir, 'modified_images'))
    preprocessor_fn = PreprocessorFactory(model_name).factory()
    postprocessor_fn = PostprocessorFactory(model_name).factory()
    inference_fn = InferenceFactory(model_name).factory()

    to_be_explained = get_images_to_explain(explain_mode, raw_image_path,
                                            num_images)

    for count, data in enumerate(to_be_explained):
        raw_image_path = data["image"]
        image_index = data["image_index"]
        model = get_model(model_name)
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
        if save_detections:
            detections_result_dir = os.path.join(result_dir, 'detections')
            if not os.path.exists(detections_result_dir):
                os.makedirs(detections_result_dir)
            write_image(os.path.join(
                detections_result_dir, "paz_postprocess.jpg"), detection_image)
        if len(detections):
            if visualize_object_index == 'all':
                objects_to_analyze = list(range(1, len(detections) + 1))
            else:
                objects_to_analyze = [int(visualize_object_index)]
            explain_all_objects(
                objects_to_analyze, raw_image_path, image_size,
                preprocessor_fn, postprocessor_fn, detections, detection_image,
                interpretation_method, box_index, to_explain, result_dir,
                class_layer_name, reg_layer_name, visualize_box_offset,
                model_name, merge_method, image_index, save_explanations,
                analyze_each_maps, ap_curve_linspace, eval_flip,
                eval_ap_explain, merge_saliency_maps, explain_top5_backgrounds,
                save_modified_images, image_adulteration_method,
                evaluate_random_map)
        else:
            LOGGER.info("No detections to analyze.")

    end_time = time.time()
    memory_profile_in_mb = process.memory_info().rss / 1024 ** 2
    LOGGER.info('Memory profiler: %s' % memory_profile_in_mb)
    LOGGER.info('Time taken: %s' % (end_time - start_time))
    LOGGER.info('%%% INTERPRETATION DONE %%%')
