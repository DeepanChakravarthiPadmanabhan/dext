import logging
import os
import numpy as np
import gin
import json
import tensorflow as tf
from tensorflow.keras import Model

from dext.dataset.coco import COCODataset
from dext.dataset.voc import VOC
from dext.factory.model_factory import ModelFactory

LOGGER = logging.getLogger(__name__)


def write_record(record, file_name, result_dir):
    file_name = os.path.join(result_dir, file_name)
    with open(file_name, 'a', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False)
        f.write(os.linesep)


def test_gpus():
    LOGGER.info("TF device check: %s" % tf.test.is_gpu_available())
    LOGGER.info("TF device name: %s" % tf.config.list_physical_devices('GPU'))


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def create_directories(result_dir, save_modified_images, save_saliency_images,
                       save_explanation_images):
    create_directory(result_dir)
    if save_modified_images:
        modified_images_dir = os.path.join(result_dir, 'modified_images')
        create_directory(modified_images_dir)
    if save_saliency_images:
        saliency_images_dir = os.path.join(result_dir, 'saliency_images')
        create_directory(saliency_images_dir)
    if save_explanation_images:
        explanation_images_dir = os.path.join(result_dir, 'explanation_images')
        create_directory(explanation_images_dir)


def get_model(model_name):
    model = ModelFactory(model_name).factory()
    return model


def build_general_custom_model(model, class_layer_name, reg_layer_name):
    if class_layer_name == reg_layer_name:
        custom_model = Model(inputs=[model.inputs], outputs=[model.output])
    else:
        custom_model = None
    return custom_model


def build_layer_custom_model(model_name, layer_name):
    model = get_model(model_name)
    custom_model = Model(inputs=[model.inputs],
                         outputs=[model.get_layer(layer_name).output])
    return custom_model


@gin.configurable
def get_images_to_explain(explain_mode, dataset_name, data_split,
                          data_split_name, raw_image_path,
                          num_images_to_explain=2, continuous_run=False,
                          result_dir=None, dataset_path=None):
    if explain_mode == 'single_image':
        data = {}
        index = (os.path.basename(
            raw_image_path)).rsplit('.jpg', 1)[0]
        data["image"] = raw_image_path
        data["image_index"] = index
        data["boxes"] = None
        to_be_explained = [data]
    else:
        if dataset_name == 'COCO':
            to_be_explained = COCODataset(dataset_path, data_split,
                                          name=data_split_name,
                                          continuous_run=continuous_run,
                                          result_dir=result_dir)
            to_be_explained = to_be_explained.load_data(
                )[:num_images_to_explain]
        elif dataset_name == 'VOC':
            to_be_explained = VOC(dataset_path, data_split, 'all',
                                  data_split_name,
                                  continuous_run=continuous_run,
                                  result_dir=result_dir)
            to_be_explained = to_be_explained.load_data(
            )[:num_images_to_explain]
        else:
            to_be_explained = None
    return to_be_explained


def get_box_arg_to_index(model_name):
    if model_name:
        box_arg_to_index = {'x_min': 0, 'y_min': 1, 'x_max': 2, 'y_max': 3}
    else:
        box_arg_to_index = {'y_min': 0, 'x_min': 1, 'y_max': 2, 'x_max': 3}
    return box_arg_to_index


def get_box_index_to_arg(model_name):
    if model_name:
        box_index_to_arg = {0: 'x_min', 1: 'y_min', 2: 'x_max', 3: 'y_max'}
    else:
        box_index_to_arg = {0: 'y_min', 1: 'x_min', 2: 'y_max', 3: 'x_max'}
    return box_index_to_arg


def get_box_index_order(model_name):
    box_index_order = []
    box_arg_to_index = get_box_arg_to_index(model_name)
    for i in BOX_ORDER_TO_PLOT[1:]:
        box_index_order.append(box_arg_to_index[i])
    return box_index_order


BOX_ORDER_TO_PLOT = ['class', 'x_min', 'y_min', 'x_max', 'y_max']


def get_explaining_info_lists(
        visualize_object_index, explaining, class_layer_name,
        reg_layer_name, box_offset, box_arg_to_index, model_name):
    object_index_list = []
    if explaining == 'Classification and Box offset':
        object_index_list.append(visualize_object_index)
        index = object_index_list[-1]
        index_list = [index, ] * 4
        object_index_list += index_list
        explaining_list = ['Classification', 'Boxoffset',
                           'Boxoffset', 'Boxoffset', 'Boxoffset']
        layer_name_list = [class_layer_name, reg_layer_name,
                           reg_layer_name, reg_layer_name,
                           reg_layer_name]
        box_index_order = get_box_index_order(model_name)
        box_offset_list = [None, ] + box_index_order
    elif explaining == 'Classification':
        object_index_list.append(visualize_object_index)
        explaining_list = ['Classification', ] * len(object_index_list)
        layer_name_list = [class_layer_name, ] * len(object_index_list)
        box_offset_list = [None, ] * len(object_index_list)
    else:
        object_index_list.append(visualize_object_index)
        explaining_list = ['Boxoffset', ] * len(object_index_list)
        layer_name_list = [reg_layer_name, ] * len(object_index_list)
        box_offset_list = [box_arg_to_index[box_offset],
                           ] * len(object_index_list)

    return object_index_list, explaining_list, layer_name_list, box_offset_list


def get_explaining_info(visualize_object_index, box_index,
                        explaining, class_layer_name, reg_layer_name,
                        box_offset, model_name):
    box_arg_to_index = get_box_arg_to_index(model_name)
    if 'None' in class_layer_name:
        class_layer_name = None
    if 'None' in reg_layer_name:
        reg_layer_name = None
    if visualize_object_index == 0:
        # Object count from 1
        visualize_object_index = visualize_object_index + 1

    visualize_object_index = visualize_object_index - 1
    explaining_info = get_explaining_info_lists(
        visualize_object_index, explaining, class_layer_name,
        reg_layer_name, box_offset, box_arg_to_index, model_name)
    object_index_list = explaining_info[0]
    explaining_list = explaining_info[1]
    layer_name_list = explaining_info[2]
    box_offset_list = explaining_info[3]

    return object_index_list, explaining_list, layer_name_list, box_offset_list


def get_box_feature_index(box_index, explaining, visualize_object, model_name,
                          visualize_box_offset=1):
    if explaining == 'Classification':
        selection = (0,
                     int(box_index[visualize_object][0]),
                     int(box_index[visualize_object][1]) + 4)
    else:
        selection = (0,
                     int(box_index[visualize_object][0]),
                     int(visualize_box_offset))
    LOGGER.info('Selected visualizing index: %s' % (selection,))
    return selection


def get_interest_index(box_index, visualize_object):
    feature_map_position = int(box_index[visualize_object][0])
    class_arg = int(box_index[visualize_object][1])
    return feature_map_position, class_arg


def resize_box(box2D, old_size, new_size):
    image_h, image_w, _ = old_size
    new_h, new_w = new_size
    x_min, y_min, x_max, y_max = box2D.coordinates
    x_min = int((x_min / image_w) * new_w)
    y_min = int((y_min / image_h) * new_h)
    x_max = int((x_max / image_w) * new_w)
    y_max = int((y_max / image_h) * new_h)
    return x_min, y_min, x_max, y_max


def resize_boxes(boxes2D, old_size, new_size):
    new_boxes = []
    for box2D in boxes2D:
        x_min, y_min, x_max, y_max = resize_box(box2D, old_size, new_size)
        new_boxes.append([x_min, y_min, x_max, y_max])
    return new_boxes


def get_saliency_mask(saliency, threshold=0.6):
    mask_2d = np.zeros(saliency.shape).astype('uint8')
    mask_2d[saliency > threshold] = 1
    return mask_2d
