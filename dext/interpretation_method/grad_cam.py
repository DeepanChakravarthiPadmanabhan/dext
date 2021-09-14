import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from paz.backend.image import resize_image
from dext.model.functional_models import get_functional_model

LOGGER = logging.getLogger(__name__)


class GradCAM:
    def __init__(self, model, model_name, image, explainer,
                 layer_name=None, visualize_idx=None,
                 preprocessor_fn=None, image_size=512,
                 grad_cam_layer=None):
        self.model = model
        self.model_name = model_name
        if "EFFICIENTDET" in self.model_name:
            self.model = get_functional_model(
                self.model_name, self.model)
        else:
            self.model = self.model
        self.image = image
        self.explainer = explainer
        self.layer_name = layer_name
        self.visualize_idx = visualize_idx
        self.preprocessor_fn = preprocessor_fn
        self.image_size = image_size
        self.image = self.check_image_size(self.image, self.image_size)
        self.image = self.preprocess_image(self.image, self.image_size)
        self.grad_cam_layer = grad_cam_layer
        if self.layer_name is None:
            self.layer_name = self.find_target_layer()
        if self.grad_cam_layer is None:
            self.grad_cam_layer = self.layer_name
        LOGGER.info('GradCAM visualization of the layer: %s'
                    % self.grad_cam_layer)
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

    def preprocess_image(self, image, image_size):
        preprocessed_image = self.preprocessor_fn(image, image_size)
        if type(preprocessed_image) == tuple:
            input_image, image_scales = preprocessed_image
        else:
            input_image = preprocessed_image
        return input_image

    def build_custom_model(self):

        if self.visualize_idx:
            custom_model = Model(
                inputs=[self.model.inputs],
                outputs=[self.model.get_layer(self.grad_cam_layer).output,
                         self.model.output[self.visualize_idx[0],
                                           self.visualize_idx[1],
                                           self.visualize_idx[2]]])
        else:
            custom_model = Model(
                inputs=[self.model.inputs],
                outputs=[self.model.get_layer(self.grad_cam_layer).output,
                         self.model.get_layer(self.layer_name).output])

        return custom_model

    def get_saliency_map(self):
        """GradCAM method for visualizing input saliency."""
        with tf.GradientTape() as tape:
            inputs = tf.cast(self.image, tf.float32)
            tape.watch(inputs)
            conv_outs, predictions = self.custom_model(inputs)
        loss = predictions
        LOGGER.debug('Conv outs from custom model: ', predictions)
        grads = tape.gradient(loss, conv_outs)
        norm_grads = tf.divide(grads, tf.reduce_mean(tf.square(grads))
                               * tf.constant(1e-5))
        weights = tf.reduce_mean(norm_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, conv_outs), axis=-1)
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)
        cam = resize_image(cam[0], (self.image_size, self.image_size))
        cam = np.stack((cam,) * 3, axis=-1)
        cam = cam[np.newaxis]
        saliency = np.asarray(cam)
        return saliency


def GradCAMExplainer(model, model_name, image, layer_name, visualize_index,
                     preprocessor_fn, image_size, grad_cam_layer=None):
    explainer = GradCAM(model, model_name, image, "GradCAM", layer_name,
                        visualize_index, preprocessor_fn, image_size,
                        grad_cam_layer)
    saliency = explainer.get_saliency_map()
    return saliency
