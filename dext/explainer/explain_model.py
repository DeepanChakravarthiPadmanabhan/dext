import logging
import os
from copy import deepcopy
import pandas as pd
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
from dext.explainer.check_saliency_maps import check_saliency
from dext.explainer.utils import get_model_class_name
from dext.explainer.postprocess_saliency import merge_saliency
from dext.explainer.analyze_saliency_maps import get_object_ap_curve


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


def explain_single_object(
        raw_image, image_size, preprocessor_fn, postprocessor_fn, detections,
        detection_image, interpretation_method, object_arg, box_index,
        to_explain, class_layer_name, reg_layer_name, visualize_box_offset,
        model_name, image_index):
    explaining_info = get_explaining_info(
        object_arg, box_index, to_explain,
        class_layer_name, reg_layer_name, visualize_box_offset,
        model_name)
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
        LOGGER.info("Generating saliency: Object - %s, "
                    "Confidence - %s, Explaining - %s, Box offset - %s"
                    % (class_name, class_confidence, explaining,
                       box_offset))
        saliency = get_single_saliency(
            interpretation_method, box_index, explaining, object_index,
            box_offset, get_model(model_name), model_name, raw_image,
            layer_name, preprocessor_fn, image_size)

        saliency_list.append(saliency)
        confidence_list.append(class_confidence)
        class_name_list.append(class_name)

        # if explaining == 'Classification' and box_offset == None:
        #     from dext.postprocessing.saliency_visualization import \
        #         plot_single_saliency
        #     plot_single_saliency(detection_image, raw_image,
        #                          saliency, class_confidence,
        #                          class_name, explaining,
        #                          interpretation_method, model_name)

        # analyze saliency maps
        metrics = analyze_saliency_maps(
            detections, raw_image, saliency, object_index)
        saliency_iou = metrics[0]
        saliency_centroid = metrics[1]
        saliency_variance = metrics[2]
        df_metrics_entry = [
            str(image_index), object_index, explaining,
            detections[object_index], saliency_iou, saliency_centroid,
            saliency_variance]

    combined_saliency = merge_saliency(saliency_list)
    ap_curve = get_object_ap_curve(
        combined_saliency, raw_image, preprocessor_fn, postprocessor_fn,
        image_size, model_name, image_index)
    df_ap_curve_entry = [str(image_index), object_arg, ap_curve]

    f = plot_all(detection_image, raw_image, saliency_list,
                 confidence_list, class_name_list, explaining_list,
                 box_offset_list, to_explain, interpretation_method,
                 model_name, "subplot")

    # saving results
    f.savefig('explanation_' + str(image_index) + "_" + "obj"
              + str(object_arg) + '.jpg')
    LOGGER.info("Box and class labels, after post-processing: %s"
                % box_index)
    return df_metrics_entry, df_ap_curve_entry


def explain_all_objects(
        objects_to_analyze, raw_image, image_size, preprocessor_fn,
        postprocessor_fn, detections, detection_image, interpretation_method,
        box_index, to_explain, class_layer_name, reg_layer_name,
        visualize_box_offset, model_name, image_index, df_metrics_image,
        df_ap_curve_image):
    for object_arg in objects_to_analyze:
        df_metrics_entry, df_ap_curve_entry = explain_single_object(
            raw_image, image_size, preprocessor_fn, postprocessor_fn,
            detections, detection_image, interpretation_method,
            object_arg, box_index, to_explain, class_layer_name,
            reg_layer_name, visualize_box_offset, model_name, image_index)
        df_metrics_image.loc[len(df_metrics_image)] = df_metrics_entry
        df_ap_curve_image.loc[len(df_ap_curve_image)] = df_ap_curve_entry
    return df_metrics_image, df_ap_curve_image



def explain_model(model_name, explain_mode, raw_image_path,
                  image_size=512, class_layer_name=None,
                  reg_layer_name=None,
                  to_explain="Classification",
                  interpretation_method="IntegratedGradients",
                  visualize_object_index=None, visualize_box_offset=1,
                  num_images=2):
    model = get_model(model_name)
    preprocessor_fn = PreprocessorFactory(model_name).factory()
    postprocessor_fn = PostprocessorFactory(model_name).factory()
    inference_fn = InferenceFactory(model_name).factory()

    to_be_explained = get_images_to_explain(explain_mode, raw_image_path,
                                            num_images)
    df_metrics_columns = ["image_index", "object_index",
                          "explaining", "detection", "saliency_iou",
                          "saliency_centroid", "saliency_variance"]
    df_ap_curve_columns = ["image_index", "object_index", "ap_50percent"]
    df_metrics = pd.DataFrame(columns=df_metrics_columns)
    df_ap_curve = pd.DataFrame(columns=df_ap_curve_columns)

    for count, data in enumerate(to_be_explained):
        raw_image = data["image"]
        image_index = data["image_index"]
        loader = LoadImage()
        raw_image = loader(raw_image)
        raw_image = raw_image.astype('uint8')
        image = deepcopy(raw_image)
        LOGGER.info("Explanation module input image ID: %s"
                    % str(image_index))

        # forward pass - get model outputs for input image
        forward_pass_outs = inference_fn(
            model, image, preprocessor_fn,
            postprocessor_fn, image_size)
        detection_image = forward_pass_outs[0]
        detections = forward_pass_outs[1]
        box_index = forward_pass_outs[2]
        write_image('images/results/paz_postprocess.jpg', detection_image)
        LOGGER.info("Detections: %s" % detections)

        if len(detections):
            if visualize_object_index == 'all':
                objects_to_analyze = list(range(1, len(detections) + 1))
            else:
                objects_to_analyze = [int(visualize_object_index)]
            df_metrics, df_ap_curve = explain_all_objects(
                objects_to_analyze, raw_image, image_size, preprocessor_fn,
                postprocessor_fn, detections, detection_image,
                interpretation_method, box_index, to_explain,
                class_layer_name, reg_layer_name, visualize_box_offset,
                model_name, image_index, df_metrics, df_ap_curve)
        else:
            LOGGER.info("No detections to analyze.")

    excel_writer = pd.ExcelWriter(
        os.path.join('images/results', "report.xlsx"),
        engine="xlsxwriter")
    df_metrics.to_excel(excel_writer, sheet_name="metrics")
    df_ap_curve.to_excel(excel_writer, sheet_name="ap_curve")
    excel_writer.save()
