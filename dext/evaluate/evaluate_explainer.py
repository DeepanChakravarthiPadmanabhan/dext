import os
import logging
import numpy as np
import json

from paz.abstract.messages import Box2D
from dext.evaluate.explainer_metrics import get_metrics
from dext.evaluate.explainer_metrics import merge_all_maps
from dext.explainer.utils import get_model
from memory_profiler import profile

LOGGER = logging.getLogger(__name__)


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def load_data(results_dir, num_images):
    if os.path.exists(results_dir):
        file = os.path.join(results_dir, 'saliency_image_paths')
        data = [json.loads(line) for line in open(file, 'r')]
        data = np.array(data)
        uniques = list(np.unique(data[:, 0]))[:num_images]
        data_selected = []
        for n, i in enumerate(uniques):
            image_level_data = data[data[:, 0] == i]
            if type(num_images) == int:
                object_ids = list(np.unique(image_level_data[:, 1]))[
                             :num_images]
            else:
                object_ids = list(np.unique(image_level_data[:, 1]))
            for j in object_ids:
                object_level_data = image_level_data[
                    image_level_data[:, 1] == j]
                data_selected.append(object_level_data)
        return data_selected
    else:
        raise ValueError('Results directory not found.')


def evaluate_image_index(sequence, model_name, image_size, ap_curve_linspace,
                         result_dir, explain_top5_backgrounds,
                         save_modified_images, image_adulteration_method,
                         eval_flip, eval_ap_explain, merge_saliency_maps,
                         merge_method, coco_result_file, model):
    column_names = ['image_index', 'object_index', 'box', 'confidence',
                    'class', 'explaining', 'boxoffset', 'saliency_path',
                    'image_path']
    image_index = int(sequence[0][column_names.index('image_index')])
    image_path = sequence[0][column_names.index('image_path')]
    object_index = int(sequence[0][column_names.index('object_index')])
    detection = Box2D(sequence[0][column_names.index('box')],
                      sequence[0][column_names.index('confidence')],
                      sequence[0][column_names.index('class')])
    saliency_list = []
    for i in sequence:
        LOGGER.info('Evaluating sequence: %s' % i)
        saliency_path = i[column_names.index('saliency_path')]
        explaining = i[column_names.index('explaining')]
        boxoffset = i[column_names.index('boxoffset')]
        saliency = np.load(saliency_path)
        saliency_list.append(saliency)
        get_metrics(detection, image_path, saliency, image_index, object_index,
                    boxoffset, model_name, image_size, explaining,
                    ap_curve_linspace, result_dir, explain_top5_backgrounds,
                    save_modified_images, image_adulteration_method, eval_flip,
                    eval_ap_explain, coco_result_file, model)
    if merge_saliency_maps:
        merge_all_maps(detection, image_path, saliency_list, image_index,
                       object_index, 'combined', model_name, image_size,
                       'combined', ap_curve_linspace, result_dir,
                       explain_top5_backgrounds, save_modified_images,
                       image_adulteration_method, eval_flip, eval_ap_explain,
                       merge_method, coco_result_file, model)

@profile
def evaluate_explainer(model_name, interpretation_method, image_size,
                       results_dir, num_images, ap_curve_linspace,
                       eval_flip, eval_ap_explain, merge_saliency_maps,
                       merge_method, save_modified_images, coco_result_file,
                       image_adulteration_method, explain_top5_backgrounds):
    results_dir = os.path.join(results_dir,
                               model_name + '_' + interpretation_method)
    data = load_data(results_dir, num_images)
    model = get_model(model_name)
    for n, sequence in enumerate(batch(data, 1)):
        evaluate_image_index(sequence[0], model_name, image_size,
                             ap_curve_linspace, results_dir,
                             explain_top5_backgrounds, save_modified_images,
                             image_adulteration_method, eval_flip,
                             eval_ap_explain, merge_saliency_maps,
                             merge_method, coco_result_file, model)
