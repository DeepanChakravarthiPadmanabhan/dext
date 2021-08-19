import numpy as np
import tensorflow as tf

from paz.backend.image.opencv_image import write_image
from paz.backend.image import resize_image

from dext.model.efficientdet.efficientdet import EFFICIENTDETD0
from dext.model.efficientdet.utils import raw_images, efficientdet_preprocess
from dext.model.efficientdet.efficientdet_postprocess import efficientdet_postprocess
from dext.method.integrated_gradient import IntegratedGradients
from dext.postprocessing.visualization import visualize_saliency_grayscale, plot_all

def get_interest_idx(idx):

    interest_neuron_index = idx[0][0]
    interest_category_index = idx[0][1]

    return interest_neuron_index, interest_category_index

def get_visualize_idx(idx, class_outputs, box_outputs):

    interest_neuron_index, interest_category_index = get_interest_idx(idx)
    level_num_boxes = []
    for level in box_outputs:
        level_num_boxes.append(level.shape[0] * level.shape[1] * level.shape[2] * 9)

    sum_all = []
    for n, i in enumerate(level_num_boxes):
        sum_all.append(sum(level_num_boxes[:n + 1]))

    bp_level = 0
    remaining_idx = interest_neuron_index
    for n, i in enumerate(sum_all):
        if i < interest_neuron_index:
            bp_level = n + 1
            remaining_idx = interest_neuron_index - i

    print("selections: ", bp_level, remaining_idx, sum_all, interest_category_index, interest_neuron_index)
    selected_class_level = class_outputs[bp_level].numpy()
    selected_class_level = np.ones((1, selected_class_level.shape[1],
                                    selected_class_level.shape[2], 9, 90))
    selected_class_level_reshaped = selected_class_level.reshape((1, -1, 90))

    print('BOX SHAPES CLASS: ', selected_class_level.shape, selected_class_level_reshaped.shape)

    interest_neuron_class = np.unravel_index(
        np.ravel_multi_index((0, int(remaining_idx), interest_category_index),
                             selected_class_level_reshaped.shape), selected_class_level.shape)

    selected_box_level = box_outputs[bp_level].numpy()
    selected_box_level = np.ones((1, selected_box_level.shape[1],
                                    selected_box_level.shape[2], 9, 4))
    selected_box_level_reshaped = selected_box_level.reshape((1, -1, 4))
    print('BOX SHAPES BOX: ', selected_box_level.shape, selected_box_level_reshaped.shape)

    interest_neuron_box = np.unravel_index(
        np.ravel_multi_index((0, int(remaining_idx), 1),
                             selected_box_level_reshaped.shape), selected_box_level.shape)


    bp_class_h = interest_neuron_class[1]
    bp_class_w = interest_neuron_class[2]
    bp_class_index = interest_neuron_class[4] + (interest_neuron_class[3] * 90)

    bp_box_h = interest_neuron_box[1]
    bp_box_w = interest_neuron_box[2]
    bp_box_index = interest_neuron_box[4] + (interest_neuron_box[3] * 4)

    print("PREDICTED BOX - CLASS: ", (bp_level, bp_class_h, bp_class_w, bp_class_index))
    print("PREDICTED BOX - BOX: ", (bp_level, bp_box_h, bp_box_w, bp_box_index))


    return (bp_level, bp_class_h, bp_class_w, bp_class_index)

def efficientdet_ig_explainer():

    model = EFFICIENTDETD0()
    image_size = model.image_size
    input_image, image_scales = efficientdet_preprocess(raw_images, image_size)
    resized_raw_image = resize_image(raw_images, (image_size, image_size))

    # Functional API calling only provides access to intermediate tensors
    original_dim = (image_size, image_size, 3)
    original_inputs = tf.keras.Input(shape=(original_dim), name="input")
    branch_tensors = model.backbone(original_inputs, False, True)
    feature_levels = branch_tensors[model.min_level:model.max_level + 1]
    # Build additional input features that are not from backbone.
    for resample_layer in model.resample_layers:
        feature_levels.append(resample_layer(
            feature_levels[-1], False, None))
    # BiFPN layers
    fpn_features = model.fpn_cells(feature_levels, False)
    # Classification head
    class_outputs = model.class_net(fpn_features, False)
    # Box regression head
    box_outputs = model.box_net(fpn_features, False)
    efdt = tf.keras.Model(inputs=original_inputs, outputs=[class_outputs, box_outputs])
    class_outputs, box_outputs = efdt(input_image)
    efdt.summary()

    image, detections, class_map_idx = efficientdet_postprocess(
        model, class_outputs, box_outputs, image_scales, raw_images)

    l, h, w, idx = get_visualize_idx(class_map_idx, class_outputs, box_outputs)
    baseline = np.zeros(shape=(1, model.image_size, model.image_size, raw_images.shape[-1]))
    m_steps = 2
    ig = IntegratedGradients(efdt, baseline, layer_name='class_net',
                             visualize_idx=(l, h, w, idx))
    ig_attributions = ig.integrated_gradients(
        image=resized_raw_image, m_steps=m_steps, batch_size=1)

    saliency = visualize_saliency_grayscale(ig_attributions)

    f = plot_all(image, resized_raw_image, saliency[0])
    f.savefig('explanation.jpg')
    l, h, w, idx = get_visualize_idx(class_map_idx, class_outputs, box_outputs)
    write_image('images/results/paz_postprocess.jpg', image)
    print(detections)
    print('To match class idx: ', class_map_idx)

efficientdet_ig_explainer()
