from dext.model.efficientdet.efficientdet_postprocess import efficientdet_postprocess
import tensorflow as tf
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

from dext.model.efficientdet.efficientdet import EFFICIENTDETD0
from dext.model.efficientdet.utils import raw_images, efficientdet_preprocess
from paz.backend.image.opencv_image import write_image

class IntegratedGradients:
    def __init__(self, model, layer_name=-1, visualize_idx=None):
        """
        Model: pre-softmax layer (logit layer)
        :param model:
        :param layer_name:
        """
        self.model = model
        self.layer_name = layer_name
        self.visualize_idx = visualize_idx

        if self.layer_name == None:
            self.layer_name = self.find_target_layer()
        self.igModel = self.build_ig_model()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply Integrated Gradients.")


    def build_ig_model(self):
        if self.visualize_idx:
            igModel = Model(
                inputs=[self.model.inputs],
                outputs=[self.model.get_layer(self.layer_name).output[:, self.visualize_idx - 1], self.model.output],
            )
        else:
            igModel = Model(
                inputs=[self.model.inputs],
                outputs=[self.model.get_layer(self.layer_name).output, self.model.output],
            )
        if 'class' in self.layer_name:
            igModel.get_layer(self.layer_name).activation = None
        return igModel


    def interpolate_images(self,
                           baseline,
                           image,
                           alphas):
        alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
        baseline_x = baseline
        input_x = image
        delta = input_x - baseline_x
        images = baseline_x + alphas_x * delta
        return images

    def compute_gradients(self, image):
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            tape.watch(inputs)
            conv_outs, preds = self.igModel(inputs)

        print('Conv outs shape: ', conv_outs.shape)
        grads = tape.gradient(conv_outs, inputs)
        return grads

    def integral_approximation(self, gradients):
        grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
        integrated_gradients = tf.math.reduce_mean(grads, axis=0)
        return integrated_gradients

    def get_normalized_interpolated_images(self, interpolated_images):
        image_size = (interpolated_images[0].shape[0], self.model.image_size,
                      self.model.image_size, interpolated_images[0].shape[-1])
        new_interpolated_image = []
        for i in interpolated_images:
            normimage, _ = efficientdet_preprocess(i, image_size)
            new_interpolated_image.append(normimage)
        new_interpolated_image = tf.concat(new_interpolated_image, axis=0)
        print("interpolated size: ", new_interpolated_image.shape)
        return new_interpolated_image

    def integrated_gradients(self,
                             baseline,
                             image,
                             m_steps,
                             batch_size):
        # 1. Generate alphas.
        alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps + 1)

        # Initialize TensorArray outside loop to collect gradients.
        gradient_batches = tf.TensorArray(tf.float32, size=m_steps + 1)

        # Iterate alphas range and batch computation for speed, memory efficient, and scaling to larger m_steps
        for alpha in tf.range(0, len(alphas), batch_size):
            from_ = alpha
            to = tf.minimum(from_ + batch_size, len(alphas))
            alpha_batch = alphas[from_: to]

            # 2. Generate interpolated inputs between baseline and input.
            interpolated_path_input_batch = self.interpolate_images(baseline=baseline,
                                                                    image=image,
                                                                    alphas=alpha_batch)


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
        integrated_gradients = (image - baseline) * avg_gradients

        return integrated_gradients

    def plot_attributions(self,
                          image,
                          attributions,
                          save_path):
        # Sum of the attributions across color channels for visualization.
        # The attribution mask shape is a grayscale image with height and width
        # equal to the original image.
        attribution_mask = tf.reduce_sum(tf.math.abs(ig_attributions), axis=-1)
        fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(12, 12))

        axs[0, 0].set_title('Baseline image')
        axs[0, 0].imshow(baseline[0].numpy().astype('uint8'))
        axs[0, 0].axis('off')

        axs[0, 1].set_title('Original image')
        axs[0, 1].imshow(image[0].astype('uint8'))
        axs[0, 1].axis('off')

        axs[1, 0].set_title('Attribution mask')
        axs[1, 0].imshow(attribution_mask[0].numpy(), cmap=plt.cm.inferno)
        axs[1, 0].axis('off')

        axs[1, 1].set_title('Overlay')
        axs[1, 1].imshow(attribution_mask[0].numpy(), cmap=plt.cm.inferno)
        axs[1, 1].imshow(image[0].astype('uint8'), alpha=0.4)
        axs[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(save_path)


if __name__ == "__main__":

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
    for i in box_outputs:
        print(i.shape)
    for i in class_outputs:
        print(i.shape)
    inp = tf.keras.layers.Input((1, image_size, image_size, 3))
    # model = tf.keras.Model(inputs=inp, outputs=model.output)
    baseline = tf.zeros(shape=(1, model.image_size, model.image_size, raw_images.shape[-1]))
    m_steps = 2
    ig = IntegratedGradients(model, layer_name='box_net')
    ig_attributions = ig.integrated_gradients(baseline=baseline,
                                              image=raw_images,
                                              m_steps=m_steps,
                                              batch_size=1,
                                              )
    ig.plot_attributions(image, ig_attributions, 'a.jpg')

    for i in model.layers:
        print(i.name)
    print(model.get_layer('box_net').output)






