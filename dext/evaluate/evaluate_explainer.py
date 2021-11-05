import os
import logging
import numpy as np
import json

LOGGER = logging.getLogger(__name__)


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def load_data(results_dir):
    if os.path.exists(results_dir):
        file = os.path.join(results_dir, 'saliency_image_paths')
        data = [json.loads(line) for line in open(file, 'r')]
        data = np.array(data)
        return data
    else:
        raise ValueError('Results directory not found.')


def evaluate_explainer(model_name, interpretation_method, image_size,
                       result_path, results_dir, num_images, ap_curve_linspace,
                       eval_flip, eval_ap_explain, merge_saliency_maps,
                       merge_method):
    results_dir = os.path.join(results_dir,
                               model_name + '_' + interpretation_method)
    data = load_data(results_dir)
    # Columns in data
    # column_names = ['image_index', 'object_index', 'box', 'confidence',
    #                 'class', 'explaining', 'boxoffset', 'saliency_path',
    #                 'image_path']
    for sequence in batch(data, 1):
        pass





