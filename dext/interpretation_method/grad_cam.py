import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from paz.backend.image import resize_image
from dext.abstract.explanation import Explainer
from dext.postprocessing.saliency_visualization import (
    visualize_saliency_grayscale)
from dext.explainer.utils import get_model

LOGGER = logging.getLogger(__name__)


class GradCAM(Explainer):
    def __init__(self, model, model_name, image, explainer,
                 layer_name=None, visualize_idx=None,
                 preprocessor_fn=None, image_size=512,
                 grad_cam_layer=None, guided_grad_cam=True):
        super().__init__(model, model_name, image, explainer)
        LOGGER.info('STARTING GRADCAM')
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
        self.grad_cam_layer = grad_cam_layer
        if self.layer_name is None:
            self.layer_name = self.find_target_layer()
        if type(self.grad_cam_layer) == int:
            self.grad_cam_layer = self.model.layers[self.grad_cam_layer].name
        if self.grad_cam_layer is None:
            self.grad_cam_layer = self.layer_name
        LOGGER.info('GradCAM visualization of the layer: %s'
                    % self.grad_cam_layer)
        self.guided_grad_cam = guided_grad_cam
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
        grads = grads[0]
        conv_outs = conv_outs[0]
        # TODO: Debug Guided GradCAM. Currently only GradCAM works.
        if self.guided_grad_cam:
            gate_f = tf.cast(conv_outs > 0, 'float32')
            gate_r = tf.cast(grads > 0, 'float32')
            norm_grads = gate_r * gate_f * grads
        else:
            norm_grads = tf.divide(grads, tf.reduce_mean(tf.square(grads))
                                   * tf.constant(1e-5))
        weights = tf.reduce_mean(norm_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, conv_outs), axis=-1)
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)
        cam = resize_image(cam, (self.image_size, self.image_size))
        cam = np.stack((cam,) * 3, axis=-1)
        cam = cam[np.newaxis]
        saliency = np.asarray(cam)
        return saliency


def GradCAMExplainer(model_name, image, interpretation_method,
                     layer_name, visualize_index, preprocessor_fn,
                     image_size, grad_cam_layer=None, guided_grad_cam=False):
    model = get_model(model_name)
    model.summary()
    if 'SSD' in model_name:
        if visualize_index[-1] <= 3:
            grad_cam_layer = 45
        else:
            grad_cam_layer = 38
    elif 'EFFICIENTDET' in model_name:
        grad_cam_layer = None
    explainer = GradCAM(model, model_name, image, interpretation_method,
                        layer_name, visualize_index, preprocessor_fn,
                        image_size, grad_cam_layer, guided_grad_cam)
    saliency = explainer.get_saliency_map()
    saliency = visualize_saliency_grayscale(saliency, 100)
    return saliency
