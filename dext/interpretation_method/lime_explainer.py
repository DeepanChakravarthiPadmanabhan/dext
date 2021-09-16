import logging
import numpy as np
from tensorflow.keras.models import Model
from lime import lime_image
# import matplotlib.pyplot as plt
# from skimage.segmentation import  mark_boundaries
# import tensorflow as tf

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

        custom_model = Model(inputs=[self.model.inputs],
                             outputs=[self.model.output])
        return custom_model

    def batch_predict(self, images):
        images_processed = []
        for n, i in enumerate(images):
            ex_image = self.preprocess_image(i, self.image_size)
            ex_image = ex_image[0]
            images_processed.append(ex_image)
        explain_images = np.stack(images_processed, axis=0)
        output = self.custom_model(explain_images).numpy()
        if self.visualize_idx[2] <= 3:
            output = output[
                     :, self.visualize_idx[1],
                     self.visualize_idx[2]: self.visualize_idx[2] + 1]
        else:
            output = output[
                     :, self.visualize_idx[1],
                     self.visualize_idx[2]: self.visualize_idx[2] + 1]
        return output

    def get_lime_hyperparameters(self):
        num_samples = 30
        if self.visualize_idx[2] <= 3:
            min_weight = 1e-5
            num_features = 10
        else:
            min_weight = 1e-3
            num_features = 5
        hyperparameters = {"num_samples": num_samples,
                           "min_weight": min_weight,
                           "num_features": num_features}
        return hyperparameters

    def get_saliency_map(self):
        image = self.image.copy()
        prediction_fn = self.batch_predict
        explainer = lime_image.LimeImageExplainer(verbose=True)
        # TODO: Check the parameters and verify the best arguments in cluster.
        # Tunable params: num_samples=10000, num_features=5
        hyperparameters = self.get_lime_hyperparameters()
        explanation = explainer.explain_instance(
            image.astype('double'), prediction_fn, labels=np.arange(1),
            top_labels=1, hide_color=0.0,
            num_samples=hyperparameters["num_samples"], batch_size=8,
            random_seed=10)
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=True,
            num_features=hyperparameters["num_features"],
            min_weight=hyperparameters["min_weight"], hide_rest=False)
        # Mask of important pixels
        # plt.imsave("marked.jpg", mask)
        # Overlap with important pixel boundary marked
        # plt.imsave("temp.jpg", mark_boundaries(temp, mask).astype('uint8'))
        return mask.astype('float32')


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
