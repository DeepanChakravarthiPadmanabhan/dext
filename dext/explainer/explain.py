import numpy as np
import tensorflow as tf

from paz.backend.image.opencv_image import write_image
from paz.backend.image import resize_image

from dext.model.efficientdet.efficientdet import EFFICIENTDETD0
from dext.model.efficientdet.utils import raw_images, efficientdet_preprocess
from dext.model.efficientdet.efficientdet_postprocess import efficientdet_postprocess
from dext.method.integrated_gradient import IntegratedGradients

from dext.postprocessing.visualization import visualize_saliency_grayscale, plot_all


def get_visualize_idx(idx, class_outputs, box_outputs):
    class_1 = class_outputs[0].numpy()
    class_1_re = class_1.reshape(1, -1, 90)
    print(class_1.shape, class_1_re.shape)
    ge = np.unravel_index(np.ravel_multi_index((0, 290, 38), class_1_re.shape), class_1.shape)

    # box_1 = box_outputs[0].numpy()
    # box_1_re = box_1.reshape(-1, 4)
    # print(box_1_re.shape, box_1.shape)
    # ge = np.unravel_index(np.ravel_multi_index((289, 0), box_1_re.shape), box_1.shape)
    # l = 0
    # h = ge[0]
    # w = ge[1]
    # idx = ge[2]

    l = 0
    h = 56
    w = 56
    idx = 221

    return (l, h, w, idx)

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

    print(l, h, w, idx)
    write_image('images/results/paz_postprocess.jpg', image)
    print(detections)
    print('To match class idx: ', class_map_idx)

efficientdet_ig_explainer()
