import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from dext.abstract.explanation import Explainer
from dext.model.utils import get_all_layers
from dext.postprocessing.saliency_visualization import (
    visualize_saliency_grayscale)
from dext.explainer.utils import get_model
from dext.utils.get_image import get_image

LOGGER = logging.getLogger(__name__)


@tf.custom_gradient
def guided_relu(x):
    def grad(dy):
        return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy
    return tf.nn.relu(x), grad


class GuidedBackpropagation(Explainer):
    def __init__(self, model, model_name, image, explainer,
                 layer_name=None, visualize_idx=None,
                 preprocessor_fn=None, image_size=512):
        super().__init__(model, model_name, image, explainer)
        LOGGER.info('STARTING GUIDED BACKPROPAGATION')
        self.model = model
        self.model_name = model_name
        self.image = image
        self.explainer = explainer
        self.layer_name = layer_name
        self.visualize_idx = visualize_idx
        self.preprocessor_fn = preprocessor_fn
        self.image_size = image_size
        self.image = self.check_image_size(self.image, self.image_size)
        self.image = self.preprocess_image(self.image, self.image_size)
        if self.layer_name is None:
            self.layer_name = self.find_target_layer()
        self.custom_model = self.build_custom_model()

    def check_image_size(self, image, image_size):
        if image.shape != (image_size, image_size, 3):
            image, _ = self.preprocessor_fn(image, image_size, True)
            if len(image.shape) != 3:
                image = image[0]
        return image

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError(
            "Could not find 4D layer. Cannot apply guided backpropagation.")

    def preprocess_image(self, image, image_size):
        input_image, _ = self.preprocessor_fn(image, image_size)
        return input_image

    def build_custom_model(self):
        if self.visualize_idx:
            custom_model = Model(
                inputs=[self.model.inputs],
                outputs=[self.model.output[self.visualize_idx[0],
                                           self.visualize_idx[1],
                                           self.visualize_idx[2]]])
        else:
            custom_model = Model(
                inputs=[self.model.inputs],
                outputs=[self.model.get_layer(self.layer_name).output])

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
            inputs = tf.cast(self.image, tf.float32)
            tape.watch(inputs)
            conv_outs = self.custom_model(inputs)
        LOGGER.info('Conv outs from custom model: %s' % conv_outs)
        grads = tape.gradient(conv_outs, inputs)
        saliency = np.asarray(grads)
        return saliency


def GuidedBackpropagationExplainer(model_name, image_path,
                                   interpretation_method, layer_name,
                                   visualize_index, preprocessor_fn,
                                   image_size):
    image = get_image(image_path)
    model = get_model(model_name)
    explainer = GuidedBackpropagation(model, model_name, image,
                                      interpretation_method, layer_name,
                                      visualize_index, preprocessor_fn,
                                      image_size)
    saliency = explainer.get_saliency_map()
    saliency_stat = (np.min(saliency), np.max(saliency))
    saliency = visualize_saliency_grayscale(saliency)
    return saliency, saliency_stat
