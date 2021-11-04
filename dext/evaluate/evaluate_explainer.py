import os
import logging

LOGGER = logging.getLogger(__name__)


def evaluate_explainer(model_name, interpretation_method, image_size,
                       result_path, result_dir,
                       ap_curve_linspace, eval_flip, eval_ap_explain,
                       merge_saliency_maps, merge_method,):
    column_names = ['image_index', 'object_index', 'box', 'confidence',
                    'class', 'explaining', 'boxoffset', 'saliency_path',
                    'image_path']
    result_dir = os.path.join(result_dir,
                              model_name + '_' + interpretation_method)


