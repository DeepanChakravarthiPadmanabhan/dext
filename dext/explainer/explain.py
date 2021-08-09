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
    class_outputs, box_outputs = model(input_image)
    image, detections, class_map_idx = efficientdet_postprocess(
        model, class_outputs, box_outputs, image_scales, raw_images)
    print(detections)
    write_image('paz_postprocess.jpg', image)
    print('task completed')
    print('To match class idx: ')
    print(class_map_idx)

    inp = tf.keras.layers.Input((1, image_size, image_size, 3))
    # model = tf.keras.Model(inputs=inp, outputs=model.output)
    baseline = tf.zeros(shape=(1, model.image_size, model.image_size, raw_images.shape[-1]))
    m_steps = 2
    ig = IntegratedGradients(model, baseline, layer_name='box_net')
    ig_attributions = ig.integrated_gradients(
        image=raw_images, m_steps=m_steps, batch_size=1)
    ig.plot_attributions(image, ig_attributions, 'a.jpg')

    for i in model.layers:
        print(i.name)
    print(model.get_layer('box_net').output)


efficientdet_ig_explainer()






