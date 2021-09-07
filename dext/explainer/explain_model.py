import logging
import os
import pandas as pd
from copy import deepcopy
from paz.backend.image.opencv_image import write_image

from dext.model.model_factory import ModelFactory
from dext.model.preprocess_factory import PreprocessorFactory
from dext.model.postprocess_factory import PostprocessorFactory
from dext.interpretation_method.interpretation_method_factory import \
    ExplainerFactory
from dext.postprocessing.saliency_visualization import \
    visualize_saliency_grayscale
from dext.postprocessing.saliency_visualization import plot_all
from dext.explainer.utils import get_box_feature_index
from dext.explainer.utils import get_explaining_info
from dext.explainer.utils import get_images_to_explain
from dext.explainer.analyze_saliency_maps import analyze_saliency_maps
from dext.explainer.check_saliency_maps import check_saliency
from dext.utils.class_names import get_class_name_efficientdet
from dext.inference.inference import inference_image


LOGGER = logging.getLogger(__name__)


def explain_object(interpretation_method, box_index,
                   class_outputs, box_outputs, explaining,
                   visualize_object_index, visualize_box_offset,
                   model, model_name, raw_image, layer_name,
                   preprocessor_fn, image_size):
    # select - get index to visualize saliency input image
    box_features = get_box_feature_index(
        box_index, class_outputs, box_outputs, explaining,
        visualize_object_index, visualize_box_offset)

    # interpret - apply interpretation method
    interpretation_method_fn = ExplainerFactory(
        interpretation_method).factory()
    saliency = interpretation_method_fn(
        model, model_name, raw_image, layer_name,
        box_features, preprocessor_fn, image_size)
    return saliency


def explain_model(model_name, explain_mode, raw_image_path,
                  image_size=512, class_layer_name=None,
                  reg_layer_name=None,
                  to_explain="Classification",
                  interpretation_method="IntegratedGradients",
                  visualize_object_index=None, visualize_box_offset=1,
                  num_images=2):
    explanation_save_file = (os.path.basename(raw_image_path)).rsplit('.jpg', 1)[0]
    LOGGER.info("Explanation module input image ID: %s" % explanation_save_file)
    model_fn = ModelFactory(model_name).factory()
    model = model_fn()

    preprocessor_fn = PreprocessorFactory(model_name).factory()
    postprocessor_fn = PostprocessorFactory(model_name).factory()

    to_be_explained = get_images_to_explain(explain_mode, raw_image_path,
                                            num_images)
    df_metrics_columns = ["image_index", "object_index", "explaining", "class", "iou"]
    df_metrics = pd.DataFrame(columns=df_metrics_columns)

    for count, data in enumerate(to_be_explained):
        image, labels = data
        image = image[0].astype('uint8')

        # forward pass - get model outputs for input image
        forward_pass_outs = inference_image(
            model, image, preprocessor_fn,
            postprocessor_fn, image_size)
        detection_image = forward_pass_outs[0]
        detections = forward_pass_outs[1]
        box_index = forward_pass_outs[2]
        class_outputs = forward_pass_outs[3]
        box_outputs = forward_pass_outs[4]

        if len(detections):
            explaining_info = get_explaining_info(
                visualize_object_index, box_index, to_explain,
                class_layer_name, reg_layer_name, visualize_box_offset)

            print(explaining_info)
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
                class_name = get_class_name_efficientdet('COCO')[
                    box_index[object_index][1]]
                class_confidence = box_index[object_index][2]
                LOGGER.info("Generating saliency: Object - %s, "
                            "Confidence - %s, Explaining - %s" %
                            (class_name, class_confidence, explaining))
                saliency = explain_object(
                    interpretation_method, box_index, class_outputs,
                    box_outputs, explaining, object_index,
                    box_offset, deepcopy(model), model_name,
                    image, layer_name, preprocessor_fn, image_size)

                # visualize - visualize the interpretation result
                saliency = visualize_saliency_grayscale(saliency)
                saliency_list.append(saliency)
                confidence_list.append(class_confidence)
                class_name_list.append(class_name)

                # analyze saliency maps
                iou = analyze_saliency_maps(detections, image, saliency, object_index)
                df_metrics.loc[len(df_metrics)] = [explanation_save_file,
                                                   object_index,
                                                   explaining,
                                                   class_name,
                                                   iou]

            f = plot_all(detection_image, image, saliency_list,
                         confidence_list, class_name_list, explaining_list,
                         box_offset_list, to_explain, interpretation_method,
                         model_name, "subplot")

            # saving results
            f.savefig('explanation_' + explanation_save_file + "_" +
                      str(visualize_object_index) + '.jpg')
            write_image('images/results/paz_postprocess.jpg', detection_image)
            LOGGER.info("Detections: %s" % detections)
            LOGGER.info("Box and class labels, after post-processing: %s" % box_index)

            excel_writer = pd.ExcelWriter(
                os.path.join('images/results', "report.xlsx"), engine="xlsxwriter")
            df_metrics.to_excel(excel_writer, sheet_name="micro")
            excel_writer.save()

            # check saliency maps
            check_saliency(model, model_name, image, preprocessor_fn,
                           postprocessor_fn, image_size, saliency_list[0], box_index)



        else:
            LOGGER.info("No detections to analyze.")
