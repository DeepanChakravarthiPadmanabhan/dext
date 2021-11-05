import logging

from dext.evaluate.analyze_saliency_maps import analyze_saliency_maps
from dext.explainer.postprocess_saliency import merge_saliency
from dext.evaluate.analyze_saliency_maps import eval_object_ap_curve
from dext.evaluate.analyze_saliency_maps import eval_numflip_maxprob_regerror
from dext.explainer.utils import write_record

LOGGER = logging.getLogger(__name__)


def get_metrics(detection, raw_image_path, saliency, image_index, object_index,
                boxoffset, model_name, image_size, explaining,
                ap_curve_linspace, result_dir, explain_top5_backgrounds,
                save_modified_images, image_adulteration_method, eval_flip,
                eval_ap_explain, coco_result_file, model=None):
    if eval_flip:
        LOGGER.info('Max probability and regression error metrics for '
                    'explaining %s & box offset %s' % (explaining, boxoffset))
        eval_metrics = analyze_saliency_maps(
            detection, raw_image_path, saliency)
        iou, centroid, variance = eval_metrics
        eval_metrics = eval_numflip_maxprob_regerror(
            saliency, raw_image_path, detection, image_size, model_name,
            ap_curve_linspace, explain_top5_backgrounds,
            save_modified_images, image_adulteration_method, result_dir, model)
        num_pixels_flipped, max_prob_curve, reg_error_curve = eval_metrics

        df_class_flip_entry = [str(image_index), object_index, explaining,
                               boxoffset, detection.coordinates,
                               detection.score, detection.class_name,
                               iou, centroid, variance, num_pixels_flipped]
        write_record(df_class_flip_entry, 'class_flip', result_dir)

        df_max_prob_entry = [str(image_index), object_index,
                             num_pixels_flipped, explaining, boxoffset, ]
        df_max_prob_entry = df_max_prob_entry + max_prob_curve
        write_record(df_max_prob_entry, 'max_prob', result_dir)

        df_reg_error_entry = [str(image_index), object_index,
                              num_pixels_flipped, explaining, boxoffset, ]
        df_reg_error_entry = df_reg_error_entry + reg_error_curve
        write_record(df_reg_error_entry, 'reg_error', result_dir)

    if eval_ap_explain:
        LOGGER.info('AP Curve metrics for explaining %s & box offset %s'
                    % (explaining, boxoffset))
        ap_curve = eval_object_ap_curve(
            saliency, raw_image_path,
            image_size, model_name, image_index, ap_curve_linspace,
            explain_top5_backgrounds, save_modified_images,
            image_adulteration_method, result_dir, coco_result_file, model)
        df_ap_curve_entry = [str(image_index), object_index,
                             explaining, boxoffset, ]
        df_ap_curve_entry = df_ap_curve_entry + ap_curve
        write_record(df_ap_curve_entry, 'ap_curve', result_dir)


def merge_all_maps(detection, image_path, saliency_list, image_index,
                   object_index, boxoffset, model_name, image_size, explaining,
                   ap_curve_linspace, result_dir, explain_top5_backgrounds,
                   save_modified_images, image_adulteration_method, eval_flip,
                   eval_ap_explain, merge_method, coco_result_file,
                   model=None):
    combined_map = merge_saliency(saliency_list, merge_method)
    get_metrics(detection, image_path, combined_map, image_index, object_index,
                boxoffset, model_name, image_size, explaining,
                ap_curve_linspace, result_dir, explain_top5_backgrounds,
                save_modified_images, image_adulteration_method, eval_flip,
                eval_ap_explain, coco_result_file, model)
