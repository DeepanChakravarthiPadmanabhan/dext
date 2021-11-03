import logging

from dext.explainer.analyze_saliency_maps import analyze_saliency_maps
from dext.explainer.postprocess_saliency import merge_saliency
from dext.explainer.analyze_saliency_maps import eval_object_ap_curve
from dext.explainer.analyze_saliency_maps import eval_numflip_maxprob_regerror
from dext.explainer.utils import write_record

LOGGER = logging.getLogger(__name__)


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
            save_modified_images, image_adulteration_method, result_dir)
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
            image_adulteration_method, result_dir)
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
