import tensorflow as tf

from paz.backend.image.opencv_image import write_image

from dext.model.efficientdet.efficientdet import EFFICIENTDETD0
from dext.model.efficientdet.utils import raw_images, efficientdet_preprocess
from dext.model.efficientdet.efficientdet_postprocess import efficientdet_postprocess
from dext.method.integrated_gradient import IntegratedGradients



# TODO: Working on this function.
def efficientdet_ig_explainer():

    model = EFFICIENTDETD0()
    image_size = model.image_size
    input_image, image_scales = efficientdet_preprocess(raw_images, image_size)

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
    # Subclassing model does not give access to tensors
    # outputs = model(original_inputs)
    # class_outputs, box_outputs = outputs

    image, detections, class_map_idx = efficientdet_postprocess(
        model, class_outputs, box_outputs, image_scales, raw_images)
    print(detections)
    write_image('images/results/paz_postprocess.jpg', image)
    print('task completed')
    print('To match class idx: ')
    print(class_map_idx)
    import tensorflow.keras.backend as K
    for i in model.layers:
        print(i.name)
        a = i.output
        print(a)
        for val in a:
            print(val)
            print(K.eval(val))
            print(type(val))


efficientdet_ig_explainer()

# inp = tf.keras.layers.Input((1, image_size, image_size, 3))
# # model = tf.keras.Model(inputs=inp, outputs=model.output)
# baseline = tf.zeros(shape=(1, model.image_size, model.image_size, raw_images.shape[-1]))
# m_steps = 2
# ig = IntegratedGradients(model, baseline, layer_name='box_net')
# ig_attributions = ig.integrated_gradients(
#     image=raw_images, m_steps=m_steps, batch_size=1)
# ig.plot_attributions(image, ig_attributions, 'a.jpg')






