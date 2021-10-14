import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from paz.processors.image import LoadImage
from paz.processors.image import resize_image
from dext.model.mask_rcnn.mask_rcnn_preprocess import mask_rcnn_preprocess
from dext.model.mask_rcnn.utils import norm_boxes_graph
from dext.model.mask_rcnn.config import Config
from dext.model.mask_rcnn.mask_rcnn_postprocess import mask_rcnn_postprocess
from dext.model.mask_rcnn.mask_rcnn_detection_model import mask_rcnn_detection
from dext.postprocessing.saliency_visualization import \
    visualize_saliency_grayscale


@tf.custom_gradient
def guided_relu(x):
    def grad(dy):
        return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy
    return tf.nn.relu(x), grad


class GuidedBackpropagation:
    def __init__(self, model, model_name, image, explainer,
                 layer_name=None, visualize_idx=None,
                 preprocessor_fn=None, config=None):
        self.model = model
        self.model_name = model_name
        self.image = image
        self.explainer = explainer
        self.layer_name = layer_name
        self.visualize_idx = visualize_idx
        self.preprocessor_fn = preprocessor_fn
        self.config = config
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
        normalized_images, windows = mask_rcnn_preprocess(self.config, image)
        normalized_images = tf.cast(normalized_images[0], tf.float32)
        normalized_images = tf.expand_dims(normalized_images, axis=0)
        return normalized_images

    def build_custom_model(self):
        custom_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=self.model.output[self.visualize_idx[0],
                                      self.visualize_idx[1],
                                      self.visualize_idx[2]])
        # all_layers = get_all_layers(custom_model)
        # all_layers = [act_layer for act_layer in all_layers
        #               if hasattr(act_layer, 'activation')]
        # for layer in all_layers:
        #     if layer.activation == tf.keras.activations.relu:
        #         layer.activation = guided_relu
        #
        # # To get logits without softmax
        # if 'class' in self.layer_name:
        #     custom_model.get_layer(self.layer_name).activation = None
        return custom_model

    def get_saliency_map(self):
        """Guided backpropagation method for visualizing input saliency."""
        with tf.GradientTape() as tape:
            inputs = self.normalized_images
            tape.watch(inputs)
            conv_outs = self.custom_model(inputs)
            conv_outs = tf.cast(conv_outs, tf.float32)
        print('Conv outs from custom model: %s' % conv_outs)
        grads = tape.gradient(conv_outs, inputs)
        saliency = np.asarray(grads)
        print("Saliency shape: ", saliency.shape, saliency)
        return saliency


class TestConfig(Config):
    NAME = "test"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80


def test(image, weights_path):
    config = TestConfig()
    normalized_images, window = mask_rcnn_preprocess(config, image)
    image_size = normalized_images[0].shape

    config_window = norm_boxes_graph(window[0], image_size[:2])
    config.WINDOW = config_window

    model = mask_rcnn_detection(config=config, image_size=image_size)
    model.load_weights(weights_path)
    input_image = tf.expand_dims(normalized_images[0], axis=0)
    out = model(input_image)
    out = out.numpy()
    detections, det_image = mask_rcnn_postprocess(
        image[0], normalized_images[0], window[0], out[0])
    print("Detections are here: ", detections)
    gbp = GuidedBackpropagation(model, "Mask_RCNN", image,
                                "GBP", "boxes", (0, 0, 0),
                                mask_rcnn_preprocess, config)
    saliency = gbp.get_saliency_map()

    saliency = visualize_saliency_grayscale(saliency)
    plt.imsave('saliency_mask.jpg', saliency)
    return detections, det_image


raw_image = "images/000000128224.jpg"
weights_path = 'new_weights_maskrcnn.h5'
loader = LoadImage()
raw_image = loader(raw_image)
image = raw_image.copy()
detections, det_image = test([image], weights_path)
plt.imsave('paz_maskrcnn.jpg', det_image)
print("done")
