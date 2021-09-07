import logging
import numpy as np

from paz.processors.image import LoadImage
from dext.dataset.coco_dataset import COCOGenerator


LOGGER = logging.getLogger(__name__)


def get_images_to_explain(explain_mode, raw_image_path,
                          num_images_to_explain=2):
    if explain_mode == 'single_image':
        loader = LoadImage()
        raw_image = loader(raw_image_path)
        to_be_explained = (([raw_image], None),)
    else:
        dataset_path = "/media/deepan/externaldrive1/datasets_project_repos/"
        dataset_folder = "mscoco"
        data_dir = dataset_path + dataset_folder
        to_be_explained = COCOGenerator(data_dir, "train2017",
                                        num_images_to_explain)
    return to_be_explained


def get_explaining_info(visualize_object_index, num_visualize, box_index,
                        explaining, class_layer_name, reg_layer_name,
                        box_offset):
    if num_visualize > len(box_index):
        LOGGER.info("Number of detections less than objects to visualize. "
                    "Switching to single object visualization.")
        num_visualize = 1
    if visualize_object_index == 0:
        # Object count from 1
        visualize_object_index = visualize_object_index + 1
    if visualize_object_index:
        # Visualize object index is given higher priority
        num_visualize = 1

    object_index_list = []
    if (visualize_object_index) and (num_visualize == 1):
        # Select the visualize_object index
        object_index_list.append(visualize_object_index - 1)
    else:
        # If visualize_object is none
        object_index_list = list(range(num_visualize))

    if explaining == 'Classification and Box offset':
        index = object_index_list[-1]
        index_list = [index, ] * 4
        object_index_list += index_list
        explaining_list = ['Classification', 'Box offset',
                           'Box offset', 'Box offset', 'Box offset']
        layer_name_list = [class_layer_name, reg_layer_name,
                           reg_layer_name, reg_layer_name,
                           reg_layer_name]
        box_offset_list = [None, 0, 1, 2, 3]
    elif explaining == 'Classification':
        explaining_list = ['Classification', ] * len(object_index_list)
        layer_name_list = [class_layer_name, ] * len(object_index_list)
        box_offset_list = [None, ] * len(object_index_list)
    else:
        explaining_list = ['Box offset', ] * len(object_index_list)
        layer_name_list = [reg_layer_name, ] * len(object_index_list)
        box_offset_list = [box_offset, ] * len(object_index_list)

    return object_index_list, explaining_list, layer_name_list, box_offset_list


def get_interest_index(box_index, visualize_object):
    feature_map_position = int(box_index[visualize_object][0])
    class_arg = int(box_index[visualize_object][1])
    return feature_map_position, class_arg


def get_box_feature_index(box_index, class_outputs, box_outputs,
                          explaining, visualize_object,
                          visualize_box_offset=1):
    feature_map_position, class_arg = get_interest_index(
        box_index, visualize_object)
    level_num_boxes = []
    for level in box_outputs:
        level_num_boxes.append(
            level.shape[0] * level.shape[1] * level.shape[2] * 9)

    sum_all = []
    for n, i in enumerate(level_num_boxes):
        sum_all.append(sum(level_num_boxes[:n + 1]))

    bp_level = 0
    remaining_idx = feature_map_position
    for n, i in enumerate(sum_all):
        if i < feature_map_position:
            bp_level = n + 1
            remaining_idx = feature_map_position - i

    if explaining == "Classification":
        selected_class_level = class_outputs[bp_level].numpy()
        selected_class_level = np.ones((1, selected_class_level.shape[1],
                                        selected_class_level.shape[2], 9, 90))
        selected_class_level_reshaped = selected_class_level.reshape((
            1, -1, 90))
        interest_neuron_class = np.unravel_index(
            np.ravel_multi_index((0, int(remaining_idx), class_arg),
                                 selected_class_level_reshaped.shape),
            selected_class_level.shape)
        bp_class_h = interest_neuron_class[1]
        bp_class_w = interest_neuron_class[2]
        bp_class_index = interest_neuron_class[4] + (
                interest_neuron_class[3] * 90)
        level, h, w, index = (bp_level, bp_class_h,
                              bp_class_w, bp_class_index)

    else:
        selected_box_level = box_outputs[bp_level].numpy()
        selected_box_level = np.ones((1, selected_box_level.shape[1],
                                      selected_box_level.shape[2], 9, 4))
        selected_box_level_reshaped = selected_box_level.reshape((1, -1, 4))
        interest_neuron_box = np.unravel_index(
            np.ravel_multi_index((0, int(remaining_idx), visualize_box_offset),
                                 selected_box_level_reshaped.shape),
            selected_box_level.shape)
        bp_box_h = interest_neuron_box[1]
        bp_box_w = interest_neuron_box[2]
        bp_box_index = interest_neuron_box[4] + (interest_neuron_box[3] * 4)
        level, h, w, index = (bp_level, bp_box_h,
                              bp_box_w, bp_box_index)

    return level, h, w, index


def resize_boxes(boxes2D, old_size, new_size):
    image_h, image_w, _ = old_size
    new_h, new_w = new_size
    new_boxes = []
    for box2D in boxes2D:
        x_min, y_min, x_max, y_max = box2D.coordinates
        x_min = int((x_min / image_w) * new_w)
        y_min = int((y_min / image_h) * new_h)
        x_max = int((x_max / image_w) * new_w)
        y_max = int((y_max / image_h) * new_h)
        new_boxes.append([x_min, y_min, x_max, y_max])
    return new_boxes


def get_saliency_mask(saliency, threshold=0.7):
    mask_2d = saliency.copy()
    mask_2d[np.where(mask_2d > threshold)] = 1
    mask_2d[np.where(mask_2d <= threshold)] = 0
    return mask_2d
