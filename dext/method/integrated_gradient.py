import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from dext.model.efficientdet.utils import efficientdet_preprocess
from dext.abstract.explanation import Explainer


class IntegratedGradients(Explainer):
    def __init__(self, model, image,
                 explainer="IntegreatedGradients",
                 layer_name=None, visualize_index=None,
                 steps=5, batch_size=1):
        """
        Model: pre-softmax layer (logit layer)
        :param model:
        :param layer_name:
        """
        super().__init__(model, image, explainer)
        self.model = model
        self.image = image
        self.image_size = image.shape[1]
        self.explainer = explainer
        self.layer_name = layer_name
        self.steps = steps
        self.batch_size = batch_size
        self.visualize_index = visualize_index

        self.generate_baseline()

        if self.layer_name == None:
            self.layer_name = self.find_target_layer()

        self.custom_model = self.build_custom_model()

    def generate_baseline(self):
        self.baseline = np.zeros(shape=(1, self.image.shape[0], self.image.shape[1], 3))

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply Integrated Gradients.")

    def build_custom_model(self):

        if self.visualize_index:
            custom_model = Model(
                inputs=[self.model.inputs],
                outputs=[self.model.get_layer(self.layer_name).output[
                             self.visualize_index[0]][
                             0, self.visualize_index[1],
                             self.visualize_index[2],
                             self.visualize_index[3]], self.model.output])
        else:
            custom_model = Model(
                inputs=[self.model.inputs],
                outputs=[self.model.get_layer(self.layer_name).output, self.model.output],
            )
        if 'class' in self.layer_name:
            custom_model.get_layer(self.layer_name).activation = None
        return custom_model

    def interpolate_images(self, image, alphas):
        alphas_x = alphas[:, np.newaxis, np.newaxis, np.newaxis]
        baseline_x = self.baseline
        input_x = image
        delta = input_x - baseline_x
        images = baseline_x + alphas_x * delta
        return images

    def compute_gradients(self, image):
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            tape.watch(inputs)
            conv_outs, preds = self.custom_model(inputs)

        grads = tape.gradient(conv_outs, inputs)
        return grads

    def integral_approximation(self, gradients):
        grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
        integrated_gradients = tf.math.reduce_mean(grads, axis=0)
        return integrated_gradients

    def get_normalized_interpolated_images(self, interpolated_images):
        image_size = self.baseline.shape[1]
        new_interpolated_image = []
        for i in interpolated_images:
            normimage, _ = efficientdet_preprocess(i, image_size)
            new_interpolated_image.append(normimage)
        new_interpolated_image = tf.concat(new_interpolated_image, axis=0)
        return new_interpolated_image

    def get_saliency_map(self):
        # 1. Generate alphas.
        alphas = np.linspace(start=0.0, stop=1.0, num=self.steps + 1)

        # Initialize TensorArray outside loop to collect gradients.
        gradient_batches = tf.TensorArray(tf.float32, size=self.steps + 1)

        # Iterate alphas range and batch computation for speed, memory efficient, and scaling to larger m_steps
        for alpha in tf.range(0, len(alphas), self.batch_size):
            from_ = alpha
            to = tf.minimum(from_ + self.batch_size, len(alphas))
            alpha_batch = alphas[from_: to]

            # 2. Generate interpolated inputs between baseline and input.
            interpolated_path_input_batch = self.interpolate_images(
                image=self.image, alphas=alpha_batch)

            interpolated_path_input_batch = self.get_normalized_interpolated_images(interpolated_path_input_batch)
            # 3. Compute gradients between model outputs and interpolated inputs.
            gradient_batch = self.compute_gradients(image=interpolated_path_input_batch)

            # Write batch indices and gradients to extend TensorArray.
            gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch)

        # Stack path gradients together row-wise into single tensor.
        total_gradients = gradient_batches.stack()

        # 4. Integral approximation through averaging gradients.
        avg_gradients = self.integral_approximation(gradients=total_gradients)

        # Scale integrated gradients with respect to input.
        integrated_gradients = (self.image - self.baseline) * avg_gradients

        return integrated_gradients

    def plot_attributions(self, image, ig_attributions, save_path):
        # Sum of the attributions across color channels for visualization.
        # The attribution mask shape is a grayscale image with height and width
        # equal to the original image.
        attribution_mask = tf.reduce_sum(tf.math.abs(ig_attributions), axis=-1)
        fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(12, 12))

        axs[0, 0].set_title('Baseline image')
        axs[0, 0].imshow(self.baseline[0].astype('uint8'))
        axs[0, 0].axis('off')

        axs[0, 1].set_title('Original image')
        axs[0, 1].imshow(image.astype('uint8'))
        axs[0, 1].axis('off')

        axs[1, 0].set_title('Attribution mask')
        axs[1, 0].imshow(attribution_mask[0].numpy(), cmap=plt.cm.inferno)
        axs[1, 0].axis('off')

        axs[1, 1].set_title('Overlay')
        axs[1, 1].imshow(attribution_mask[0].numpy(), cmap=plt.cm.inferno)
        axs[1, 1].imshow(image.astype('uint8'), alpha=0.4)
        axs[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(save_path)

def IntegratedGradientExplainer(model, image, layer_name, visualize_index):
    ig = IntegratedGradients(model, image, "IG", layer_name, visualize_index)
    saliency = ig.get_saliency_map()
    return saliency

