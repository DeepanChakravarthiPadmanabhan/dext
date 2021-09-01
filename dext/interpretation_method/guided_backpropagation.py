import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model


@tf.custom_gradient
def guided_relu(x):
    def grad(dy):
        return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy
    return tf.nn.relu(x), grad


class GuidedBackpropagation:
    def __init__(self, model, image, explainer, layer_name=None, visualize_idx=None):
        self.model = model
        self.image = image
        self.explainer = explainer
        self.layer_name = layer_name
        self.visualize_idx = visualize_idx

        if self.layer_name == None:
            self.layer_name = self.find_target_layer()

        self.custom_model = self.build_custom_model()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError(
            "Could not find 4D layer. Cannot apply guided backpropagation.")

    def build_custom_model(self):
        if self.visualize_idx:
            gbModel = Model(
                inputs=[self.model.inputs],
                outputs=[self.model.get_layer(self.layer_name).output[
                             self.visualize_idx[0]][
                             0, self.visualize_idx[1],
                             self.visualize_idx[2],
                             self.visualize_idx[3]]])
        else:
            gbModel = Model(
                inputs=[self.model.inputs],
                outputs=[self.model.get_layer(self.layer_name).output],
            )

        base_model_layers = [act_layer for act_layer in gbModel.layers[1].layers if hasattr(act_layer, 'activation')]
        head_layers = [
            layer for layer in gbModel.layers[1:] if hasattr(layer, "activation")
        ]
        all_layers = base_model_layers + head_layers
        for layer in all_layers:
            if layer.activation == tf.keras.activations.relu:
                layer.activation = guided_relu

        if 'class' in self.layer_name:
            gbModel.get_layer(self.layer_name).activation = None
        return gbModel

    def get_saliency_map(self):
        """Guided backpropagation method for visualizing input saliency."""
        with tf.GradientTape() as tape:
            inputs = tf.cast(self.image, tf.float32)
            tape.watch(inputs)
            conv_outs = self.custom_model(inputs)
        grads = tape.gradient(conv_outs, inputs)
        saliency = np.asarray(grads)
        return saliency

def GuidedBackpropagationExplainer(model, image, layer_name, visualize_index):
    explainer = GuidedBackpropagation(model, image, "GBP", layer_name, visualize_index)
    saliency = explainer.get_saliency_map()
    return saliency