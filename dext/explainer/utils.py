import logging
import os
import numpy as np
import gin

from dext.dataset.coco import COCODataset
from dext.factory.model_factory import ModelFactory
from dext.utils.class_names import get_class_name_efficientdet
from paz.datasets.utils import get_class_names

LOGGER = logging.getLogger(__name__)


def get_model(model_name, image=None, image_size=512):
    model = ModelFactory(model_name, image, image_size).factory()
    return model


def get_model_class_name(model_name, dataset_name):
    if "EFFICIENTDET" in model_name:
        class_names = get_class_name_efficientdet(dataset_name)
    elif "SSD" in model_name:
        class_names = get_class_names(dataset_name)
    elif model_name == "FasterRCNN":
        class_names = get_class_names(dataset_name)
    else:
        raise ValueError("Model not implemented %s" % model_name)
    return class_names


@gin.configurable
def get_images_to_explain(explain_mode, raw_image_path,
                          num_images_to_explain=2, dataset_path=None):
    if explain_mode == 'single_image':
        data = {}
        index = (os.path.basename(
            raw_image_path)).rsplit('.jpg', 1)[0]
        data["image"] = raw_image_path
        data["image_index"] = index
        data["boxes"] = None
        to_be_explained = [data]
    else:
        to_be_explained = COCODataset(dataset_path, "val", name="val2017",)
        to_be_explained = to_be_explained.load_data()[:num_images_to_explain]
    return to_be_explained


def get_box_arg_to_index(model_name):
    if 'EFFICIENTDET' in model_name or model_name == 'FasterRCNN':
        box_arg_to_index = {'y_min': 0, 'x_min': 1, 'y_max': 2, 'x_max': 3}
    else:
        box_arg_to_index = {'x_min': 0, 'y_min': 1, 'x_max': 2, 'y_max': 3}
    return box_arg_to_index


def get_box_index_to_arg(model_name):
    if 'EFFICIENTDET' in model_name or model_name == 'FasterRCNN':
        box_index_to_arg = {0: 'y_min', 1: 'x_min', 2: 'y_max', 3: 'x_max'}
    else:
        box_index_to_arg = {0: 'x_min', 1: 'y_min', 2: 'x_max', 3: 'y_max'}
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
        explaining_list = ['Classification', 'Box offset',
                           'Box offset', 'Box offset', 'Box offset']
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
        explaining_list = ['Box offset', ] * len(object_index_list)
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
    if ('SSD' in model_name) or ('EFFICIENTDET' in model_name):
        models_with_all_box_outs = True
    else:
        models_with_all_box_outs = False  # For FasterRCNN
    if models_with_all_box_outs:
        if explaining == 'Classification':
            selection = (0,
                         int(box_index[visualize_object][0]),
                         int(box_index[visualize_object][1]) + 4)
        else:
            selection = (0,
                         int(box_index[visualize_object][0]),
                         int(visualize_box_offset))
    else:
        if explaining == 'Classification':
            selection = (0,
                         int(visualize_object),
                         5)
        else:
            selection = (0,
                         int(visualize_object),
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
