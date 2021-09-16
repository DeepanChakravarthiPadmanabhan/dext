import logging
import numpy as np
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import shap
# import tensorflow as tf
# tf.compat.v1.disable_v2_behavior()
# from tensorflow.compat.v1.keras.backend import get_session

from paz.backend.image import resize_image
from paz.backend.image.opencv_image import load_image
from dext.utils.class_names import get_class_name_efficientdet
from dext.model.functional_models import get_functional_model
from dext.dataset.coco import COCODataset

LOGGER = logging.getLogger(__name__)
COCO_DATASET_PATH = '/media/deepan/externaldrive1/datasets_project_repos/coco'
# COCO_DATASET_PATH = '/scratch/dpadma2s/coco/'


class GradientSHAP:
    def __init__(self, model, model_name, image, explainer,
                 layer_name=None, visualize_idx=None,
                 preprocessor_fn=None, image_size=512,
                 num_background_images=5):
        self.model = model
        self.model_name = model_name
        self.image = image
        self.explainer = explainer
        self.layer_name = layer_name
        self.visualize_idx = visualize_idx
        self.preprocessor_fn = preprocessor_fn
        self.image_size = image_size
        self.num_background_images = num_background_images
        self.image = self.check_image_size(self.image, self.image_size)
        if self.layer_name is None:
            self.layer_name = self.find_target_layer()
        self.custom_model = self.build_custom_model()
        self.image = self.preprocess_image(self.image, self.image_size)
        self.background_images = self.get_background_images(
            self.image_size, self.num_background_images)

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

    def get_background_images(self, image_size, num_background_images):
        data_manager = COCODataset(COCO_DATASET_PATH, 'train', 'train2017')
        dataset = data_manager.load_data()
        images = []
        if len(dataset) >= num_background_images:
            dataset = dataset[:num_background_images]
        for i in dataset:
            image = load_image(i["image"])
            image = resize_image(image, (image_size, image_size))
            image = self.preprocess_image(image, image_size)[0]
            images.append(image)
        background_images = np.stack(images, axis=0)
        return background_images

    def map2layer(self, model, x, layer):
        feed_dict = dict(zip([model.layers[0].input], [x.copy()]))
        return get_session().run(model.layers[layer].input, feed_dict)

    def build_custom_model(self):
        if "EFFICIENTDET" in self.model_name:
            self.model = get_functional_model(
                self.model_name, self.model)
        else:
            self.model = self.model
        if self.visualize_idx[2] <= 3:
            custom_model = Model(inputs=[self.model.inputs],
                                 outputs=[self.model.output[:,
                                          self.visualize_idx[1],
                                          self.visualize_idx[2]:
                                          self.visualize_idx[2] + 1]])
        else:
            custom_model = Model(inputs=[self.model.inputs],
                                 outputs=[self.model.output[:,
                                          self.visualize_idx[1],
                                          self.visualize_idx[2]:
                                          self.visualize_idx[2] + 1]])
        return custom_model

    def get_saliency_map(self):
        image = self.image.copy().astype('float32')
        e = shap.GradientExplainer(
            (self.custom_model.layers[-1].input,
             self.custom_model.layers[-1].output),
            self.map2layer(self.custom_model, image.ref(), -1),
            local_smoothing=0
        )
        shap_values, indexes = e.shap_values(
            self.map2layer(self.custom_model, image, "boxes"),
            ranked_outputs=1)

        index_names = np.vectorize(
            lambda x: get_class_name_efficientdet("COCO")[x])(indexes)
        print(shap_values)
        shap.image_plot(shap_values, image, index_names)
        plt.savefig('GradientExplainer_shap_image_plot.jpg')
        return 1


def SHAP_GradientExplainer(model, model_name, image, interpretation_method,
                           layer_name, visualize_index, preprocessor_fn,
                           image_size, num_background_images=5):
    explainer = GradientSHAP(model, model_name, image, interpretation_method,
                             layer_name, visualize_index, preprocessor_fn,
                             image_size, num_background_images)
    saliency = explainer.get_saliency_map()
    return saliency
