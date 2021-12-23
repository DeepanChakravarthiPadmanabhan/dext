import os
import time
import logging
import numpy as np
import json
import psutil

from paz.abstract.messages import Box2D
from dext.evaluator.explainer_metrics import get_metrics
from dext.evaluator.explainer_metrics import merge_all_maps
from dext.explainer.utils import get_model
from dext.utils.select_image_ids_coco import filter_image_ids
from dext.utils.select_image_ids_coco import get_history_file
from memory_profiler import profile

LOGGER = logging.getLogger(__name__)


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def get_data(results_dir, filename):
    file = get_history_file(results_dir, filename)
    data = [json.loads(line) for line in open(file, 'r')]
    return data


def get_detection_level_maps(all_image_index, data):
    data_selected = []
    for n, i in enumerate(all_image_index):
        image_level_data = data[data[:, 0] == i]
        object_ids = list(np.unique(image_level_data[:, 1]))
        for j in object_ids:
            object_level_data = image_level_data[
                image_level_data[:, 1] == j]
            data_selected.append(object_level_data)
    return data_selected


def get_eval_image_index(results_dir, data, num_images, continuous_run):
    all_image_index = list(np.unique(data[:, 0]))
    LOGGER.info('No. of image ids: %s' % len(all_image_index))
    if continuous_run:
        load_ran_ids = filter_image_ids(results_dir, 'real_class_flip')
        LOGGER.info('Image ids already evaluated: %s' % load_ran_ids)
        all_image_index = [i for i in all_image_index if int(i) not in
                           load_ran_ids]
    LOGGER.info('No. of image ids after filtering: %s' % len(all_image_index))
    if num_images > len(all_image_index):
        raise ValueError('No. of image ids available are less.')
    all_image_index = all_image_index[:num_images]
    LOGGER.info('Images ids evaluating: %s' % all_image_index)
    return all_image_index


def load_data(results_dir, num_images, continuous_run=True):
    data = np.array(get_data(results_dir, 'saliency_image_paths'))
    all_image_index = get_eval_image_index(
        results_dir, data, num_images, continuous_run)
    data_selected = get_detection_level_maps(all_image_index, data)
    LOGGER.info('Total detections with saliency maps: %s' % len(data_selected))
    return data_selected


def evaluate_image_index(sequence, model_name, image_size, ap_curve_linspace,
                         result_dir, explain_top5_backgrounds,
                         save_modified_images, image_adulteration_method,
                         eval_deletion, eval_insertion,  eval_ap_explain,
                         merge_saliency_maps, merge_method,
                         save_all_map_metrics, coco_result_file, model):
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
        if save_all_map_metrics:
            get_metrics(detection, image_path, saliency, image_index,
                        object_index, boxoffset, model_name, image_size,
                        explaining, ap_curve_linspace, result_dir,
                        explain_top5_backgrounds, save_modified_images,
                        image_adulteration_method, eval_deletion,
                        eval_insertion, eval_ap_explain, coco_result_file,
                        model)
    if merge_saliency_maps and save_all_map_metrics:
        if merge_method == 'all':
            merge_all_maps(
                detection, image_path, saliency_list, image_index,
                object_index, 'combined_pca', model_name, image_size,
                'combined_pca', ap_curve_linspace, result_dir,
                explain_top5_backgrounds, save_modified_images,
                image_adulteration_method, eval_deletion, eval_insertion,
                eval_ap_explain, 'pca', coco_result_file, model)
            merge_all_maps(
                detection, image_path, saliency_list, image_index,
                object_index, 'combined_andavg', model_name, image_size,
                'combined_andavg', ap_curve_linspace, result_dir,
                explain_top5_backgrounds, save_modified_images,
                image_adulteration_method, eval_deletion, eval_insertion,
                eval_ap_explain, 'and_average', coco_result_file, model)
            merge_all_maps(
                detection, image_path, saliency_list, image_index,
                object_index, 'combined_oravg', model_name, image_size,
                'combined_oravg', ap_curve_linspace, result_dir,
                explain_top5_backgrounds, save_modified_images,
                image_adulteration_method, eval_deletion, eval_insertion,
                eval_ap_explain, 'or_average', coco_result_file, model)
        else:
            merge_all_maps(
                detection, image_path, saliency_list, image_index,
                object_index, 'combined', model_name, image_size,
                'combined', ap_curve_linspace, result_dir,
                explain_top5_backgrounds, save_modified_images,
                image_adulteration_method, eval_deletion, eval_insertion,
                eval_ap_explain, merge_method, coco_result_file, model)


@profile
def evaluate_explainer(model_name, interpretation_method, image_size,
                       results_dir, num_images, ap_curve_linspace,
                       eval_deletion, eval_insertion, eval_ap_explain,
                       merge_saliency_maps, merge_method, save_modified_images,
                       coco_result_file, image_adulteration_method,
                       explain_top5_backgrounds, continuous_run,
                       save_all_map_metrics):
    start_time = time.time()
    process = psutil.Process(os.getpid())
    results_dir = os.path.join(results_dir,
                               model_name + '_' + interpretation_method)
    data = load_data(results_dir, num_images, continuous_run)
    model = get_model(model_name)
    for n, sequence in enumerate(batch(data, 1)):
        LOGGER.info('Evaluating detection: %s' % n)
        evaluate_image_index(sequence[0], model_name, image_size,
                             ap_curve_linspace, results_dir,
                             explain_top5_backgrounds, save_modified_images,
                             image_adulteration_method, eval_deletion,
                             eval_insertion, eval_ap_explain,
                             merge_saliency_maps, merge_method,
                             save_all_map_metrics, coco_result_file, model)
    end_time = time.time()
    memory_profile_in_mb = process.memory_info().rss / 1024 ** 2
    LOGGER.info('Memory profiler: %s' % memory_profile_in_mb)
    LOGGER.info('Time taken: %s' % (end_time - start_time))
    LOGGER.info('%%% ANALYSIS DONE %%%')
