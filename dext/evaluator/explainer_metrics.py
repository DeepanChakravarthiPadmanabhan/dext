import logging

from dext.evaluator.analyze_saliency_maps import analyze_saliency_maps
from dext.explainer.postprocess_saliency import merge_saliency
from dext.evaluator.analyze_saliency_maps import eval_object_ap_curve
from dext.evaluator.analyze_saliency_maps import eval_realistic_errors
from dext.evaluator.analyze_saliency_maps import eval_bounded_errors
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
        eval_metrics = eval_bounded_errors(
            saliency, raw_image_path, detection, image_size, model_name,
            ap_curve_linspace, explain_top5_backgrounds,
            save_modified_images, image_adulteration_method, result_dir, model,
            object_index)
        (bound_class_conf, bound_reg_error_pixels,
         bound_reg_error_iou, bound_x_error_pixels, bound_y_error_pixels,
         bound_w_error_pixels, bound_h_error_pixels) = eval_metrics

        df_class_conf = [str(image_index), object_index, explaining,
                         boxoffset, ]
        df_class_conf = df_class_conf + bound_class_conf
        write_record(df_class_conf, 'bound_class_conf', result_dir)

        df_reg_error_pixels = [str(image_index), object_index, explaining,
                               boxoffset, ]
        df_reg_error_pixels = df_reg_error_pixels + bound_reg_error_pixels
        write_record(df_reg_error_pixels, 'bound_reg_error_pixels', result_dir)

        df_reg_error_iou = [str(image_index), object_index, explaining,
                            boxoffset, ]
        df_reg_error_iou = df_reg_error_iou + bound_reg_error_iou
        write_record(df_reg_error_iou, 'bound_reg_error_iou', result_dir)

        df_x_error_pixels = [str(image_index), object_index, explaining,
                             boxoffset, ]
        df_x_error_pixels = df_x_error_pixels + bound_x_error_pixels
        write_record(df_x_error_pixels, 'bound_x', result_dir)

        df_y_error_pixels = [str(image_index), object_index, explaining,
                             boxoffset, ]
        df_y_error_pixels = df_y_error_pixels + bound_y_error_pixels
        write_record(df_y_error_pixels, 'bound_y', result_dir)

        df_w_error_pixels = [str(image_index), object_index, explaining,
                             boxoffset, ]
        df_w_error_pixels = df_w_error_pixels + bound_w_error_pixels
        write_record(df_w_error_pixels, 'bound_w', result_dir)

        df_h_error_pixels = [str(image_index), object_index, explaining,
                             boxoffset, ]
        df_h_error_pixels = df_h_error_pixels + bound_h_error_pixels
        write_record(df_h_error_pixels, 'bound_h', result_dir)

        eval_metrics = eval_realistic_errors(
            saliency, raw_image_path, detection, image_size, model_name,
            ap_curve_linspace, explain_top5_backgrounds,
            save_modified_images, image_adulteration_method, result_dir, model)
        (num_pixels_flipped, real_class_conf, real_reg_error_pixels,
         real_reg_error_iou, real_x_error_pixels, real_y_error_pixels,
         real_w_error_pixels, real_h_error_pixels) = eval_metrics

        df_class_flip_entry = [str(image_index), object_index, explaining,
                               boxoffset, detection.coordinates,
                               detection.score, detection.class_name,
                               iou, centroid, variance, num_pixels_flipped]
        write_record(df_class_flip_entry, 'real_class_flip', result_dir)

        df_class_conf = [str(image_index), object_index,
                         num_pixels_flipped, explaining, boxoffset, ]
        df_class_conf_real = df_class_conf + real_class_conf
        write_record(df_class_conf_real, 'real_class_conf', result_dir)

        df_reg_error_pixels = [str(image_index), object_index,
                               num_pixels_flipped, explaining, boxoffset, ]
        df_reg_error_pixels = df_reg_error_pixels + real_reg_error_pixels
        write_record(df_reg_error_pixels, 'real_reg_error_pixels', result_dir)

        df_reg_error_iou = [str(image_index), object_index,
                            num_pixels_flipped, explaining, boxoffset, ]
        df_reg_error_iou = df_reg_error_iou + real_reg_error_iou
        write_record(df_reg_error_iou, 'real_reg_error_iou', result_dir)

        df_x_error_pixels = [str(image_index), object_index, explaining,
                             boxoffset, ]
        df_x_error_pixels = df_x_error_pixels + real_x_error_pixels
        write_record(df_x_error_pixels, 'real_x', result_dir)

        df_y_error_pixels = [str(image_index), object_index, explaining,
                             boxoffset, ]
        df_y_error_pixels = df_y_error_pixels + real_y_error_pixels
        write_record(df_y_error_pixels, 'real_y', result_dir)

        df_w_error_pixels = [str(image_index), object_index, explaining,
                             boxoffset, ]
        df_w_error_pixels = df_w_error_pixels + real_w_error_pixels
        write_record(df_w_error_pixels, 'real_w', result_dir)

        df_h_error_pixels = [str(image_index), object_index, explaining,
                             boxoffset, ]
        df_h_error_pixels = df_h_error_pixels + real_h_error_pixels
        write_record(df_h_error_pixels, 'real_h', result_dir)

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
