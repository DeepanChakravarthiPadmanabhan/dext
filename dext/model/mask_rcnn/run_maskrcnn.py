import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from paz.backend.image import resize_image
from paz.processors.image import LoadImage

from dext.model.mask_rcnn.mask_rcnn_preprocess import  mask_rcnn_preprocess
from dext.model.mask_rcnn.model import MaskRCNN
from dext.model.mask_rcnn.config import Config
from dext.model.mask_rcnn.utils import norm_boxes_graph
from dext.model.mask_rcnn.mask_rcnn_postprocess import mask_rcnn_postprocess
from dext.model.utils import get_all_layers
from dext.model.mask_rcnn.mask_rcnn_inference_model import inference
from dext.model.mask_rcnn.inference_graph import InferenceGraph


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
        custom_model = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.output])
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
            conv_outs = self.custom_model.predict(inputs, steps=1)
            tape.watch(inputs)
        print('Conv outs from custom model: %s' % conv_outs[0])
        grads = tape.gradient(conv_outs, inputs)
        print(grads)
        saliency = np.asarray(grads)
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

    base_model = MaskRCNN(config=config, model_dir=weights_path,
                          image_size=image_size)
    inference_model = InferenceGraph(model=base_model, config=config,
                                     include_mask=False)
    base_model.keras_model = inference_model()
    base_model.keras_model.load_weights(weights_path, by_name=True)
    model = base_model.keras_model

    detections = model.predict([normalized_images])
    results = mask_rcnn_postprocess(image[0], normalized_images[0],
                                    window[0], detections[0])
    # gbp = GuidedBackpropagation(model, "Mask_RCNN", image,
    #                             "GBP", "boxes", (0, 0, 1),
    #                             mask_rcnn_preprocess, config)
    # saliency = gbp.get_saliency_map()
    return results

raw_image = "images/000000128224.jpg"
weights_path = '/media/deepan/externaldrive1/project_repos/DEXT_versions/weights/mask_rcnn_coco.h5'

import matplotlib.pyplot as plt
loader = LoadImage()
raw_image = loader(raw_image)
image = raw_image.copy()
results, det_image = test([image], weights_path)
print(results)
plt.imsave('paz_maskrcnn.jpg', det_image)
print("done")


