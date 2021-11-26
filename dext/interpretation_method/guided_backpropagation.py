import logging
import numpy as np
import tensorflow as tf

from dext.explainer.utils import build_layer_custom_model
from dext.abstract.explanation import Explainer
from dext.model.utils import get_all_layers
from dext.postprocessing.saliency_visualization import (
    visualize_saliency_grayscale)
from dext.utils.get_image import get_image

LOGGER = logging.getLogger(__name__)


@tf.custom_gradient
def guided_relu(x):
    def grad(dy):
        return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy
    return tf.nn.relu(x), grad


class GuidedBackpropagation(Explainer):
    def __init__(self, model, model_name, image, explainer,
                 layer_name=None, visualize_index=None,
                 preprocessor_fn=None, image_size=512):
        super().__init__(model_name, image, explainer)
        LOGGER.info('STARTING GUIDED BACKPROPAGATION')
        self.model_name = model_name
        self.image = image
        self.explainer = explainer
        self.layer_name = layer_name
        self.visualize_index = visualize_index
        self.preprocessor_fn = preprocessor_fn
        self.image_size = image_size
        self.image = self.check_image_size(self.image, self.image_size)
        self.image = self.preprocess_image(self.image, self.image_size)
        self.layer_name = layer_name
        if model:
            self.custom_model = model
            self.clean_custom_model()
        else:
            self.custom_model = build_layer_custom_model(self.model_name,
                                                         self.layer_name)
            self.clean_custom_model()

    def check_image_size(self, image, image_size):
        if image.shape != (image_size, image_size, 3):
            image, _ = self.preprocessor_fn(image, image_size, True)
            if len(image.shape) != 3:
                image = image[0]
        return image

    def preprocess_image(self, image, image_size):
        input_image, _ = self.preprocessor_fn(image, image_size)
        return input_image

    def clean_custom_model(self):
        all_layers = get_all_layers(self.custom_model)
        all_layers = [act_layer for act_layer in all_layers
                      if hasattr(act_layer, 'activation')]
        for layer in all_layers:
            if layer.activation == tf.keras.activations.relu:
                layer.activation = guided_relu
        # To get logits without softmax
        if 'class' in self.layer_name:
            self.custom_model.get_layer(self.layer_name).activation = None

    def get_saliency_map(self):
        """Guided backpropagation method for visualizing input saliency."""
        with tf.GradientTape() as tape:
            inputs = tf.cast(self.image, tf.float32)
            tape.watch(inputs)
            conv_outs = self.custom_model(inputs)
            conv_outs = conv_outs[self.visualize_index[0],
                                  self.visualize_index[1],
                                  self.visualize_index[2]]
        LOGGER.info('Conv outs from custom model: %s' % conv_outs)
        grads = tape.gradient(conv_outs, inputs)
        saliency = np.asarray(grads)
        return saliency


def GuidedBackpropagationExplainer(model, model_name, image_path,
                                   interpretation_method, layer_name,
                                   visualize_index, preprocessor_fn,
                                   image_size, normalize=True):
    if isinstance(image_path, str):
        image = get_image(image_path)
    else:
        image = image_path
    explainer = GuidedBackpropagation(
        model, model_name, image, interpretation_method, layer_name,
        visualize_index, preprocessor_fn, image_size)
    saliency = explainer.get_saliency_map()
    saliency_stat = (np.min(saliency), np.max(saliency))
    if normalize:
        saliency = visualize_saliency_grayscale(saliency)
    return saliency, saliency_stat
