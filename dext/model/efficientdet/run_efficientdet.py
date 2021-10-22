import numpy as np
import tensorflow as tf
from paz.processors.image import resize_image
import matplotlib.pyplot as plt
from dext.model.efficientdet.efficientdet_model import EfficientDet
from paz.processors.image import LoadImage
from dext.model.efficientdet.utils import efficientdet_preprocess
from dext.model.efficientdet.efficientdet_postprocess import \
    efficientdet_postprocess
from dext.model.efficientdet.efficientdet_postprocess import \
    get_class_name_efficientdet
from dext.postprocessing.saliency_visualization import (
    visualize_saliency_grayscale)
from dext.postprocessing.saliency_visualization import plot_single_saliency
from dext.model.utils import get_all_layers

@tf.custom_gradient
def guided_relu(x):
    def grad(dy):
        return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy
    return tf.nn.relu(x), grad


class GuidedBackpropagation:
    def __init__(self, model, model_name, image, explainer,
                 layer_name=None, visualize_idx=None,
                 preprocessor_fn=None, image_size=512):
        self.model = model
        self.model_name = model_name
        self.image = image
        self.explainer = explainer
        self.layer_name = layer_name
        self.image_size = image_size
        self.visualize_idx = visualize_idx
        self.preprocessor_fn = preprocessor_fn
        self.normalized_images = self.preprocess_image(image)
        if self.layer_name is None:
            self.layer_name = self.find_target_layer()
        self.custom_model = self.build_custom_model()

    def check_image_size(self, image, image_size):
        if image.shape != (image_size, image_size, 3):
            image = resize_image(image, (image_size, image_size))
        return image

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError(
            "Could not find 4D layer. Cannot apply guided backpropagation.")

    def preprocess_image(self, image):
        normalized_images, windows = efficientdet_preprocess(image,
                                                             self.image_size)
        normalized_images = tf.cast(normalized_images, tf.float32)
        return normalized_images

    def build_custom_model(self):
        custom_model = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.output[self.visualize_idx[0],
                                       self.visualize_idx[1],
                                       self.visualize_idx[2]]])
        all_layers = get_all_layers(custom_model)
        all_layers = [act_layer for act_layer in all_layers
                      if hasattr(act_layer, 'activation')]
        for layer in all_layers:
            if layer.activation == tf.keras.activations.relu:
                layer.activation = guided_relu
        # To get logits without softmax
        if 'class' in self.layer_name:
            custom_model.get_layer(self.layer_name).activation = None
        return custom_model

    def get_saliency_map(self):
        """Guided backpropagation method for visualizing input saliency."""
        with tf.GradientTape() as tape:
            inputs = self.normalized_images
            tape.watch(inputs)
            conv_outs = self.custom_model(inputs)
        print('Conv outs from custom model: %s' % conv_outs)
        grads = tape.gradient(conv_outs, inputs)
        saliency = np.asarray(grads)
        return saliency

class_names = get_class_name_efficientdet('COCO')
raw_image = "images/000000128224.jpg"
loader = LoadImage()
raw_image = loader(raw_image)
image = raw_image.copy()
image, image_scales = efficientdet_preprocess(image, 512)
weights_path = "/media/deepan/externaldrive1/project_repos/" \
               "DEXT_versions/weights/paz_efficientdet_weights/"
model = EfficientDet(num_classes=90, base_weights='COCO', head_weights='COCO',
                   input_shape=(512, 512, 3), fpn_num_filters=64,
                   fpn_cell_repeats=3, box_class_repeats=3, anchor_scale=4.0,
                   min_level=3, max_level=7, fpn_weight_method='fastattention',
                   return_base=False, model_name='efficientdet-d0',
                   backbone='efficientnet-b0', weights_path=weights_path)
all_layers = get_all_layers(model)
outputs = model(image)
detection_image, detections, class_map_idx = efficientdet_postprocess(
    model, outputs, image_scales, raw_image)
print(detections)
plt.imsave("efficientdet.jpg", detection_image)
explain_object = 1
visualize_idx = (0,
                 int(class_map_idx[explain_object][0]),
                 int(class_map_idx[explain_object][1]) + 4)
print("class map idx: ", class_map_idx)
print('visualizer: ', visualize_idx)
gbp = GuidedBackpropagation(model, "EFFICIENTDETD0", raw_image, "IG",
                            "boxes", visualize_idx, efficientdet_preprocess,
                            512)
saliency = gbp.get_saliency_map()
saliency = visualize_saliency_grayscale(saliency)
plt.imsave('saliency_mask.jpg', saliency)
print('saliency.shape', saliency.shape, type(saliency))
fig = plot_single_saliency(detection_image, raw_image, saliency,
                           class_map_idx[explain_object][2],
                           class_names[class_map_idx[explain_object][1]],
                           explaining='Classification',
                           interpretation_method='IG',
                           model_name='EfficientDet')
fig.savefig("efficientdet_saliency.jpg")
