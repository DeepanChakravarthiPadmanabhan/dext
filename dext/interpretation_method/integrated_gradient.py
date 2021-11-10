import logging
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from dext.abstract.explanation import Explainer
from dext.postprocessing.saliency_visualization import (
    visualize_saliency_grayscale)
from dext.explainer.utils import build_layer_custom_model
from dext.utils.get_image import get_image

LOGGER = logging.getLogger(__name__)


class IntegratedGradients(Explainer):
    def __init__(self, model, model_name, image,
                 explainer="IntegratedGradients",
                 layer_name=None, visualize_index=None,
                 preprocessor_fn=None, image_size=512,
                 steps=5, batch_size=1):
        super().__init__(model_name, image, explainer)
        LOGGER.info('STARTING INTEGRATED GRADIENTS')
        self.model_name = model_name
        self.image = image
        self.image_size = image_size
        self.explainer = explainer
        self.layer_name = layer_name
        self.steps = steps
        self.batch_size = batch_size
        self.visualize_index = visualize_index
        self.preprocessor_fn = preprocessor_fn
        self.image = self.check_image_size(self.image, self.image_size)
        self.baseline = self.generate_baseline()
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

    def generate_baseline(self):
        baseline = np.zeros(shape=(1, self.image_size,
                            self.image_size, 3))
        return baseline

    def preprocess_image(self, image, image_size):
        input_image, _ = self.preprocessor_fn(image, image_size)
        return input_image

    def clean_custom_model(self):
        # To get logits without softmax
        if 'class' in self.layer_name:
            self.custom_model.get_layer(self.layer_name).activation = None

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
            conv_outs = self.custom_model(inputs)
            conv_outs = conv_outs[self.visualize_index[0],
                                  self.visualize_index[1],
                                  self.visualize_index[2]]
        LOGGER.info('Conv outs from custom model: %s' % conv_outs)
        grads = tape.gradient(conv_outs, inputs)
        return grads

    def integral_approximation(self, gradients):
        grads = (gradients[:-1] + gradients[1:]) / 2
        integrated_gradients = np.mean(grads, axis=0)
        return integrated_gradients

    def get_normalized_interpolated_images(self, interpolated_images):
        image_size = self.baseline.shape[1]
        new_interpolated_image = []
        for i in interpolated_images:
            normimage = self.preprocess_image(i, image_size)
            new_interpolated_image.append(normimage)
        new_interpolated_image = np.concatenate(new_interpolated_image, axis=0)
        return new_interpolated_image

    def get_saliency_map(self):
        # 1. Generate alphas.
        alphas = np.linspace(start=0.0, stop=1.0, num=self.steps + 1)
        gradient_batches = []

        # Iterate alphas range and batch computation for speed,
        # memory efficient, and scaling to larger m_steps
        for alpha in range(0, len(alphas), self.batch_size):
            LOGGER.info('Performing IG for alpha: %s' % alpha)
            from_ = alpha
            to = np.minimum(from_ + self.batch_size, len(alphas))
            alpha_batch = alphas[from_: to]

            # 2. Generate interpolated inputs between baseline and input.
            interpolated_path_input_batch = self.interpolate_images(
                image=self.image, alphas=alpha_batch)

            interpolated_path_input_batch = (
                self.get_normalized_interpolated_images(
                    interpolated_path_input_batch))
            # 3. Compute gradients between model outputs
            # and interpolated inputs.
            gradient_batch = self.compute_gradients(
                image=interpolated_path_input_batch)

            gradient_batches.append(gradient_batch)

        # Stack path gradients together row-wise into single tensor.
        total_gradients = np.concatenate(gradient_batches)

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
        fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False,
                                figsize=(12, 12))

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


def IntegratedGradientExplainer(model, model_name, image_path, interpretation_method,
                                layer_name, visualize_index, preprocessor_fn,
                                image_size, steps=2, batch_size=1):
    image = get_image(image_path)
    ig = IntegratedGradients(model, model_name, image, interpretation_method,
                             layer_name, visualize_index, preprocessor_fn,
                             image_size, steps, batch_size)
    saliency = ig.get_saliency_map()
    saliency_stat = (np.min(saliency), np.max(saliency))
    saliency = visualize_saliency_grayscale(saliency)
    return saliency, saliency_stat
