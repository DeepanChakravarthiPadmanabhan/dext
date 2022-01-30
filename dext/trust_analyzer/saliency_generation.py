import os
import logging
import matplotlib.pyplot as plt

from dext.factory.interpretation_method_factory import ExplainerFactory
from dext.explainer.utils import get_box_feature_index
from dext.explainer.utils import get_explaining_info
from dext.utils.class_names import get_classes
from dext.postprocessing.saliency_visualization import plot_saliency_human
from dext.postprocessing.saliency_visualization import plot_detection_human
from dext.explainer.utils import write_record

LOGGER = logging.getLogger(__name__)


def get_single_saliency(
        interpretation_method, box_index, explaining, visualize_object_index,
        visualize_box_offset, model_name, raw_image_path, layer_name,
        preprocessor_fn, image_size, custom_model, prior_boxes):
    # select - get index to visualize saliency input image
    box_features = get_box_feature_index(
        box_index, explaining, visualize_object_index, model_name,
        visualize_box_offset)
    # interpret - apply interpretation method
    interpretation_method_fn = ExplainerFactory(
        interpretation_method).factory()
    saliency = interpretation_method_fn(
        custom_model, model_name, raw_image_path, interpretation_method,
        layer_name, box_features, preprocessor_fn, image_size,
        prior_boxes=prior_boxes, explaining=explaining)
    return saliency


def save_saliency_record(
        result_dir, image_index, object_index, coordinates, score, class_name,
        explaining, box_offset, save_path, raw_image_path, model_name,
        interpretation_method):
    record = [image_index, object_index, coordinates, score, class_name,
              explaining, box_offset, save_path, raw_image_path,
              model_name, interpretation_method]
    write_record(record, "saliency_image_paths", result_dir)


def explain_single_object(raw_image_path, image_size, preprocessor_fn,
                          detections, interpretation_method, box_index,
                          result_dir, explaining_info, model_name, image_index,
                          class_name, class_confidence, custom_model,
                          prior_boxes):
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
            preprocessor_fn, image_size, custom_model, prior_boxes)

        save_saliency_record(
            result_dir, str(image_index), str(object_index),
            detections[object_index].coordinates, class_confidence,
            class_name, explaining, box_offset, result_dir, raw_image_path,
            model_name, interpretation_method)
        save_dir = os.path.join(result_dir,
                                model_name + '_' + interpretation_method)

        image_id = str(image_index)
        det_id = str(object_index)
        sal_type = explaining + '_' + (
            str(box_offset) if box_offset else 'None')
        det_save_name = ('detid_' + det_id + '_imageid_' + image_id + '.jpg')
        det_save_name = os.path.join(save_dir, det_save_name)
        sal_save_name = ('detid_' + det_id + '_saltype_' + sal_type +
                         '_imageid_' + image_id + '.jpg')
        sal_save_name = os.path.join(save_dir, sal_save_name)

        detection_selected = detections[object_index]
        det_fig = plot_detection_human(raw_image_path, [detection_selected])
        det_fig.savefig(det_save_name)
        LOGGER.info('Saved detection: %s ' % det_save_name)
        det_fig.clear()
        plt.close(det_fig)

        sal_fig = plot_saliency_human(raw_image_path, saliency, model_name)
        sal_fig.savefig(sal_save_name)
        LOGGER.info('Saved saliency: %s ' % sal_save_name)
        sal_fig.clear()
        plt.close(sal_fig)


def explain_all_objects(objects_to_analyze, raw_image_path, image_size,
                        preprocessor_fn, detections,
                        interpretation_method, box_index, to_explain,
                        result_dir, class_layer_name, reg_layer_name,
                        visualize_box_offset, model_name, image_index,
                        custom_model, prior_boxes,
                        dataset_name):
    for object_arg in objects_to_analyze:
        explaining_info = get_explaining_info(
            object_arg, box_index, to_explain, class_layer_name,
            reg_layer_name,
            visualize_box_offset, model_name)
        class_name = get_classes(dataset_name, model_name)[
            box_index[explaining_info[0][0]][1]]
        class_confidence = box_index[explaining_info[0][0]][2]
        LOGGER.info('Explaining - image index: %s, confidence: %s, class: %s'
                    % (str(image_index), class_confidence, class_name))
        LOGGER.info("Information - %s" % (explaining_info,))
        explain_single_object(
            raw_image_path, image_size, preprocessor_fn, detections,
            interpretation_method, box_index, result_dir, explaining_info,
            model_name, image_index, class_name, class_confidence,
            custom_model, prior_boxes)
