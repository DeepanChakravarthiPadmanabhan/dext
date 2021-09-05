import numpy as np


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

    # print("selections: ", bp_level, remaining_idx, sum_all,
    #       interest_category_index, interest_neuron_index)
    selected_class_level = class_outputs[bp_level].numpy()
    selected_class_level = np.ones((1, selected_class_level.shape[1],
                                    selected_class_level.shape[2], 9, 90))
    selected_class_level_reshaped = selected_class_level.reshape((1, -1, 90))

    # print('BOX SHAPES CLASS: ', selected_class_level.shape,
    #       selected_class_level_reshaped.shape)

    interest_neuron_class = np.unravel_index(
        np.ravel_multi_index((0, int(remaining_idx), class_arg),
                             selected_class_level_reshaped.shape),
        selected_class_level.shape)

    selected_box_level = box_outputs[bp_level].numpy()
    selected_box_level = np.ones((1, selected_box_level.shape[1],
                                  selected_box_level.shape[2], 9, 4))
    selected_box_level_reshaped = selected_box_level.reshape((1, -1, 4))
    # print('BOX SHAPES BOX: ', selected_box_level.shape,
    #       selected_box_level_reshaped.shape)

    interest_neuron_box = np.unravel_index(
        np.ravel_multi_index((0, int(remaining_idx), visualize_box_offset),
                             selected_box_level_reshaped.shape),
        selected_box_level.shape)

    # print("INTEREST NEURON CLASS: ", interest_neuron_class)
    # print("INTEREST NEURON BOX: ", interest_neuron_box)

    bp_class_h = interest_neuron_class[1]
    bp_class_w = interest_neuron_class[2]
    bp_class_index = interest_neuron_class[4] + (interest_neuron_class[3] * 90)

    bp_box_h = interest_neuron_box[1]
    bp_box_w = interest_neuron_box[2]
    bp_box_index = interest_neuron_box[4] + (interest_neuron_box[3] * 4)

    # print("PREDICTED BOX - CLASS: ", (bp_level, bp_class_h,
    #                                   bp_class_w, bp_class_index))
    # print("PREDICTED BOX - BOX: ", (bp_level, bp_box_h,
    #                                 bp_box_w, bp_box_index))

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
