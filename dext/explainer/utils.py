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


def get_explain_index(visualize_object, num_visualize, box_index):
    if num_visualize > len(box_index):
        LOGGER.info("Number of detections less than objects to visualize. "
                    "Switching to single object visualization.")
        num_visualize = 1
    if visualize_object == 0:
        # Object count from 1
        visualize_object = visualize_object + 1
    if visualize_object:
        # Visualize object index is given higher priority
        num_visualize = 1
    visualize_object_index = []
    if (visualize_object) and (num_visualize == 1):
        # Select the visualize_object index
        visualize_object_index.append(visualize_object - 1)
    else:
        # If visualize_object is none
        visualize_object_index = list(range(num_visualize))
    return visualize_object_index


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

    # LOGGER.info("selections: ", bp_level, remaining_idx, sum_all,
    #       interest_category_index, interest_neuron_index)
    selected_class_level = class_outputs[bp_level].numpy()
    selected_class_level = np.ones((1, selected_class_level.shape[1],
                                    selected_class_level.shape[2], 9, 90))
    selected_class_level_reshaped = selected_class_level.reshape((1, -1, 90))

    # LOGGER.info('BOX SHAPES CLASS: ', selected_class_level.shape,
    #       selected_class_level_reshaped.shape)

    interest_neuron_class = np.unravel_index(
        np.ravel_multi_index((0, int(remaining_idx), class_arg),
                             selected_class_level_reshaped.shape),
        selected_class_level.shape)

    selected_box_level = box_outputs[bp_level].numpy()
    selected_box_level = np.ones((1, selected_box_level.shape[1],
                                  selected_box_level.shape[2], 9, 4))
    selected_box_level_reshaped = selected_box_level.reshape((1, -1, 4))
    # LOGGER.info('BOX SHAPES BOX: ', selected_box_level.shape,
    #       selected_box_level_reshaped.shape)

    interest_neuron_box = np.unravel_index(
        np.ravel_multi_index((0, int(remaining_idx), visualize_box_offset),
                             selected_box_level_reshaped.shape),
        selected_box_level.shape)

    # LOGGER.info("INTEREST NEURON CLASS: ", interest_neuron_class)
    # LOGGER.info("INTEREST NEURON BOX: ", interest_neuron_box)

    bp_class_h = interest_neuron_class[1]
    bp_class_w = interest_neuron_class[2]
    bp_class_index = interest_neuron_class[4] + (interest_neuron_class[3] * 90)

    bp_box_h = interest_neuron_box[1]
    bp_box_w = interest_neuron_box[2]
    bp_box_index = interest_neuron_box[4] + (interest_neuron_box[3] * 4)

    # LOGGER.info("PREDICTED BOX - CLASS: ", (bp_level, bp_class_h,
    #                                         bp_class_w, bp_class_index))
    # LOGGER.info("PREDICTED BOX - BOX: ", (bp_level, bp_box_h,
    #                                       bp_box_w, bp_box_index))

    level, h, w, index = (None,) * 4
    if explaining == "Classification":
        level, h, w, index = (bp_level, bp_class_h,
                              bp_class_w, bp_class_index)
    elif explaining == "Box":
        level, h, w, index = (bp_level, bp_box_h,
                              bp_box_w, bp_box_index)
    else:
        pass

    return (level, h, w, index)
