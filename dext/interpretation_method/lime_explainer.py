from lime import lime_image
from skimage.segmentation import  mark_boundaries

import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from paz.backend.image import resize_image
from dext.model.functional_models import get_functional_model

LOGGER = logging.getLogger(__name__)


class LIME:
    def __init__(self, model, model_name, image, explainer,
                 layer_name=None, visualize_idx=None,
                 preprocessor_fn=None, image_size=512):
        self.model = model
        self.model_name = model_name
        self.image = image
        self.explainer = explainer
        self.layer_name = layer_name
        self.visualize_idx = visualize_idx
        self.preprocessor_fn = preprocessor_fn
        self.image_size = image_size
        self.image = self.check_image_size(self.image, self.image_size)
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

    def preprocess_image(self, image, image_size):
        preprocessed_image = self.preprocessor_fn(image, image_size)
        if type(preprocessed_image) == tuple:
            input_image, image_scales = preprocessed_image
        else:
            input_image = preprocessed_image
        return input_image

    def build_custom_model(self):
        if "EFFICIENTDET" in self.model_name:
            self.model = get_functional_model(
                self.model_name, self.model)
        else:
            self.model = self.model

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

        return custom_model

    def batch_predict(self, images):
        images_processed = []
        for i in images:
            ex_image = self.preprocess_image(i, self.image_size)
            ex_image = ex_image[0]
            images_processed.append(ex_image)
        explain_images = np.stack(images_processed, axis=0)
        return self.custom_model(explain_images)

    def get_saliency_map(self):
        pred_fn = self.batch_predict
        explainer = lime_image.LimeImageExplainer(verbose=True)
        explanation = explainer.explain_instance(self.image.astype('double'),
                                                 pred_fn, labels=np.arange(1),
                                                 top_labels=1, hide_color=0.0,
                                                 num_samples=2, batch_size=8,
                                                 random_seed=10)
        print('explanation: ', explanation)
        return 1


def LimeExplainer(model, model_name, image,
                  interpretation_method,
                  layer_name, visualize_index,
                  preprocessor_fn, image_size):
    explainer = LIME(model, model_name, image,
                     interpretation_method,
                     layer_name, visualize_index,
                     preprocessor_fn, image_size)
    saliency = explainer.get_saliency_map()
    return saliency
