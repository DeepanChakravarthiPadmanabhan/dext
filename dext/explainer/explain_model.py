import logging
import os
from copy import deepcopy
import pandas as pd
import numpy as np
from paz.backend.image.opencv_image import write_image
from paz.processors.image import LoadImage

from dext.explainer.utils import get_model
from dext.factory.preprocess_factory import PreprocessorFactory
from dext.factory.postprocess_factory import PostprocessorFactory
from dext.factory.interpretation_method_factory import \
    ExplainerFactory
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
        visualize_box_offset, model, model_name, raw_image, layer_name,
        preprocessor_fn, image_size):
    # select - get index to visualize saliency input image
    box_features = get_box_feature_index(
        box_index, explaining, visualize_object_index, visualize_box_offset)
    # interpret - apply interpretation method
    interpretation_method_fn = ExplainerFactory(
        interpretation_method).factory()
    saliency = interpretation_method_fn(
        model, model_name, raw_image, interpretation_method, layer_name,
        box_features, preprocessor_fn, image_size)
    return saliency


def get_metrics(detections, raw_image, gt_boxes, saliency, object_arg,
                preprocessor_fn, postprocessor_fn, model_name, image_size,
                explaining, ap_curve_linspace, image_index,
                explain_top5_backgrounds, df_class_flip, df_ap_curve,
                df_max_prob, df_reg_error):
    saliency_iou, saliency_centroid, saliency_variance = analyze_saliency_maps(
        detections, raw_image, saliency, object_arg)
    eval_metrics = eval_numflip_maxprob_regerror(
        saliency, raw_image, gt_boxes, detections, preprocessor_fn,
        postprocessor_fn, image_size, model_name, object_arg,
        ap_curve_linspace, explain_top5_backgrounds)
    num_pixels_flipped, max_prob_curve, reg_error_curve = eval_metrics
    df_class_flip_entry = [
        str(image_index), object_arg, explaining,
        detections[object_arg], saliency_iou, saliency_centroid,
        saliency_variance, num_pixels_flipped]
    df_class_flip.loc[len(df_class_flip)] = df_class_flip_entry
    ap_curve = eval_object_ap_curve(
        saliency, raw_image, preprocessor_fn, postprocessor_fn,
        image_size, model_name, image_index, ap_curve_linspace,
        explain_top5_backgrounds)
    df_ap_curve_entry = [str(image_index), object_arg,
                         num_pixels_flipped,
                         explaining, ]
    df_ap_curve_entry = df_ap_curve_entry + ap_curve
    df_ap_curve.loc[len(df_ap_curve)] = df_ap_curve_entry
    df_max_prob_entry = [str(image_index), object_arg,
                         num_pixels_flipped, explaining, ]
    df_max_prob_entry = df_max_prob_entry + max_prob_curve
    df_max_prob.loc[len(df_max_prob)] = df_max_prob_entry
    df_reg_error_entry = [str(image_index), object_arg,
                          num_pixels_flipped, explaining, ]
    df_reg_error_entry = df_reg_error_entry + reg_error_curve
    df_reg_error.loc[len(df_reg_error)] = df_reg_error_entry
    return df_class_flip, df_ap_curve, df_max_prob, df_reg_error


def merge_all_maps(saliency_list, merge_method, analyze_each_maps,
                   detections, raw_image, gt_boxes, preprocessor_fn,
                   postprocessor_fn, image_size, model_name,
                   ap_curve_linspace, image_index, explain_top5_backgrounds,
                   object_arg, df_class_flip, df_ap_curve, df_max_prob,
                   df_reg_error):
    combined_saliency = merge_saliency(saliency_list, merge_method)
    if analyze_each_maps:
        df_class_flip, df_ap_curve, df_max_prob, df_reg_error = get_metrics(
            detections, raw_image, gt_boxes, combined_saliency, object_arg,
            preprocessor_fn, postprocessor_fn, model_name, image_size,
            'combined', ap_curve_linspace, image_index,
            explain_top5_backgrounds, df_class_flip, df_ap_curve, df_max_prob,
            df_reg_error)
    return df_class_flip, df_ap_curve, df_max_prob, df_reg_error


def explain_single_object(
        raw_image, image_size, gt_boxes, preprocessor_fn, postprocessor_fn,
        detections, detection_image, interpretation_method, object_arg,
        box_index, to_explain, result_dir, class_layer_name, reg_layer_name,
        visualize_box_offset, model_name, merge_method, image_index,
        save_explanations, analyze_each_maps, ap_curve_linspace, df_class_flip,
        df_ap_curve, df_max_prob, df_reg_error, merge_saliency_maps,
        explain_top5_backgrounds):
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
        saliency = get_single_saliency(
            interpretation_method, box_index, explaining, object_index,
            box_offset, get_model(model_name), model_name, raw_image,
            layer_name, preprocessor_fn, image_size)

        saliency_list.append(saliency)
        confidence_list.append(class_confidence)
        class_name_list.append(class_name)
        # analyze saliency maps
        if analyze_each_maps:
            all_dfs = get_metrics(
                detections, raw_image, gt_boxes, saliency, object_index,
                preprocessor_fn, postprocessor_fn, model_name, image_size,
                explaining + str(box_offset), ap_curve_linspace, image_index,
                explain_top5_backgrounds, df_class_flip, df_ap_curve,
                df_max_prob, df_reg_error)
            df_class_flip, df_ap_curve, df_max_prob, df_reg_error = all_dfs

    if merge_saliency_maps:
        df_class_flip, df_ap_curve, df_max_prob, reg_error = merge_all_maps(
            saliency_list, merge_method, analyze_each_maps, detections,
            raw_image, gt_boxes, preprocessor_fn, postprocessor_fn, image_size,
            model_name, ap_curve_linspace, image_index,
            explain_top5_backgrounds, object_index_list[0],
            df_class_flip, df_ap_curve, df_max_prob, df_reg_error)

    if save_explanations:
        explanation_result_dir = os.path.join(result_dir, 'explanations')
        if not os.path.exists(explanation_result_dir):
            os.makedirs(explanation_result_dir)
        f = plot_all(detection_image, raw_image, saliency_list,
                     confidence_list, class_name_list, explaining_list,
                     box_offset_list, to_explain, interpretation_method,
                     model_name, "subplot")
        f.savefig(os.path.join(explanation_result_dir, 'explanation_'
                  + str(image_index) + "_" + "obj" + str(object_arg) + '.jpg'))
    LOGGER.info("Box and class labels, after post-processing: %s"
                % box_index)
    LOGGER.info("Completed explaining image index: %s" % str(image_index))
    return df_class_flip, df_ap_curve, df_max_prob, df_reg_error


def explain_all_objects(
        objects_to_analyze, raw_image, image_size, gt_boxes, preprocessor_fn,
        postprocessor_fn, detections, detection_image, interpretation_method,
        box_index, to_explain, result_dir, class_layer_name, reg_layer_name,
        visualize_box_offset, model_name, merge_method, image_index,
        save_explanations, analyze_each_maps, ap_curve_linspace, df_class_flip,
        df_ap_curve, df_max_prob, df_reg_error, merge_saliency_maps,
        explain_top5_backgrounds):
    for object_arg in objects_to_analyze:
        all_dfs = explain_single_object(
            raw_image, image_size, gt_boxes, preprocessor_fn, postprocessor_fn,
            detections, detection_image, interpretation_method,
            object_arg, box_index, to_explain, result_dir, class_layer_name,
            reg_layer_name, visualize_box_offset, model_name, merge_method,
            image_index, save_explanations, analyze_each_maps,
            ap_curve_linspace, df_class_flip, df_ap_curve, df_max_prob,
            df_reg_error, merge_saliency_maps, explain_top5_backgrounds)
        df_class_flip, df_ap_curve, df_max_prob, df_reg_error = all_dfs
    return df_class_flip, df_ap_curve, df_max_prob, df_reg_error


def explain_model(model_name, explain_mode, raw_image_path, image_size=512,
                  class_layer_name=None, reg_layer_name=None,
                  to_explain="Classification", result_dir='images/results/',
                  interpretation_method="IntegratedGradients",
                  visualize_object_index=None, visualize_box_offset=None,
                  num_images=2, merge_method='add', save_detections=False,
                  save_explanations=False, analyze_each_maps=False,
                  ap_curve_linspace=20, merge_saliency_maps=False,
                  explain_top5_backgrounds=True):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    model = get_model(model_name)
    preprocessor_fn = PreprocessorFactory(model_name).factory()
    postprocessor_fn = PostprocessorFactory(model_name).factory()
    inference_fn = InferenceFactory(model_name).factory()

    to_be_explained = get_images_to_explain(explain_mode, raw_image_path,
                                            num_images)
    df_class_flip_columns = ["image_index", "object_index", "explaining",
                             "detection", "saliency_iou", "saliency_centroid",
                             "saliency_variance", "pixels_flipped"]
    df_ap_curve_columns = ["image_index", "object_index", "pixels_flipped",
                           "explaining"]
    ap_50percent_columns = ["ap_50percent_" + str(round(n, 2))
                            for n in np.linspace(0, 1, ap_curve_linspace)]
    df_max_prob_columns = ["image_index", "object_index", "pixels_flipped",
                           "explaining"]
    max_prob_percent = ["max_prob_" + str(round(n, 2))
                        for n in np.linspace(0, 1, ap_curve_linspace)]
    df_reg_error_columns = ["image_index", "object_index", "pixels_flipped",
                            "explaining"]
    reg_error_percent = ["reg_error_" + str(round(n, 2))
                         for n in np.linspace(0, 1, ap_curve_linspace)]
    df_ap_curve_columns = df_ap_curve_columns + ap_50percent_columns
    df_max_prob_columns = df_max_prob_columns + max_prob_percent
    df_reg_error_columns = df_reg_error_columns + reg_error_percent
    df_class_flip = pd.DataFrame(columns=df_class_flip_columns)
    df_ap_curve = pd.DataFrame(columns=df_ap_curve_columns)
    df_max_prob = pd.DataFrame(columns=df_max_prob_columns)
    df_reg_error = pd.DataFrame(columns=df_reg_error_columns)

    for count, data in enumerate(to_be_explained):
        raw_image = data["image"]
        image_index = data["image_index"]
        gt_boxes = data["boxes"]
        loader = LoadImage()
        raw_image = loader(raw_image)
        raw_image = raw_image.astype('uint8')
        image = deepcopy(raw_image)
        LOGGER.info('%%% BEGIN EXPLANATION MODULE %%%')
        LOGGER.info('Explaining image count: %s' % str(count + 1))
        LOGGER.info("Explanation input image ID: %s" % str(image_index))

        # forward pass - get model outputs for input image
        forward_pass_outs = inference_fn(
            model, image, preprocessor_fn, postprocessor_fn, image_size,
            explain_top5_backgrounds)
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
            all_dfs = explain_all_objects(
                objects_to_analyze, raw_image, image_size, gt_boxes,
                preprocessor_fn, postprocessor_fn, detections, detection_image,
                interpretation_method, box_index, to_explain, result_dir,
                class_layer_name, reg_layer_name, visualize_box_offset,
                model_name, merge_method, image_index, save_explanations,
                analyze_each_maps, ap_curve_linspace, df_class_flip,
                df_ap_curve, df_max_prob, df_reg_error, merge_saliency_maps,
                explain_top5_backgrounds)
            df_class_flip, df_ap_curve, df_max_prob, df_reg_error = all_dfs
        else:
            LOGGER.info("No detections to analyze.")
    excel_writer = pd.ExcelWriter(
        os.path.join(result_dir, "report.xlsx"),
        engine="xlsxwriter")
    df_class_flip.to_excel(excel_writer, sheet_name="class_flip")
    df_ap_curve.to_excel(excel_writer, sheet_name="ap_curve")
    df_max_prob.to_excel(excel_writer, sheet_name="max_prob_curve")
    df_reg_error.to_excel(excel_writer, sheet_name="reg_error_curve")
    excel_writer.save()
    LOGGER.info('%%% INTERPRETATION DONE %%%')
