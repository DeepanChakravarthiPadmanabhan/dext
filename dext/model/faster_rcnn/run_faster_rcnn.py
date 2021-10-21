import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from paz.processors.image import LoadImage
from paz.processors.image import resize_image
from dext.model.faster_rcnn.faster_rcnn_preprocess import (
    faster_rcnn_preprocess)
from dext.model.faster_rcnn.faster_rcnn_postprocess import (
    faster_rcnn_postprocess)
from dext.model.utils import get_all_layers
from dext.postprocessing.saliency_visualization import (
    visualize_saliency_grayscale)
from dext.model.faster_rcnn.faster_rcnn_detection_model import (
    get_faster_rcnn_model)


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
        normalized_images, windows = faster_rcnn_preprocess(image,
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


def run_simple(image, weight_path, image_size):
    model = get_faster_rcnn_model(image, image_size, weight_path)
    input_image, image_scales = faster_rcnn_preprocess(image, image_size)
    out = model(input_image)
    det_image, detections, class_map_idx = faster_rcnn_postprocess(
        model, out, image_scales, image, image_size)
    print("Detections are here: ", detections)
    gbp = GuidedBackpropagation(model, "Faster_RCNN", image,
                                "GBP", "boxes", (0, 0, 0),
                                faster_rcnn_preprocess)
    saliency = gbp.get_saliency_map()
    saliency = visualize_saliency_grayscale(saliency)
    plt.imsave('saliency_mask.jpg', saliency)
    return detections, det_image


raw_image = "images/000000117156.jpg"
weight_path = "/media/deepan/externaldrive1/project_repos/"
folder = "DEXT_versions/weights/paz_faster_rcnn_weights/"
weight_path = weight_path + folder
loader = LoadImage()
raw_image = loader(raw_image)
image = raw_image.copy()
detections, det_image = run_simple(image, weight_path, 512)
plt.imsave('paz_faster_rcnn.jpg', det_image)
print("done")
