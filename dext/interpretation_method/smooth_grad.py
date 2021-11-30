import logging
import numpy as np

from dext.abstract.explanation import Explainer
from dext.interpretation_method.integrated_gradient import (
    IntegratedGradientExplainer)
from dext.interpretation_method.guided_backpropagation import (
    GuidedBackpropagationExplainer)
from dext.postprocessing.saliency_visualization import (
    visualize_saliency_grayscale)
from dext.utils.get_image import get_image

LOGGER = logging.getLogger(__name__)


class SmoothGrad(Explainer):
    def __init__(self, model, model_name, image_path,
                 explainer="SmoothGrad_IntegreatedGradients", layer_name=None,
                 visualize_index=None, preprocessor_fn=None, image_size=512,
                 standard_deviation=0.15, nsamples=5, magnitude=1, steps=5,
                 batch_size=1, prior_boxes=None, explaining=None):
        super().__init__(model_name, image_path, explainer)
        self.model = model
        self.model_name = model_name
        self.image_path = image_path
        self.image = get_image(self.image_path)
        self.original_shape = self.image.shape
        self.image_size = image_size
        self.explainer = explainer
        self.layer_name = layer_name
        self.standard_deviation = standard_deviation
        self.nsamples = nsamples
        self.magnitude = magnitude
        self.steps = steps
        self.batch_size = batch_size
        self.visualize_index = visualize_index
        self.preprocessor_fn = preprocessor_fn
        self.prior_boxes = prior_boxes
        self.explaining = explaining

    def get_saliency_map(self):
        stddev = self.standard_deviation * (np.max(self.image) -
                                            np.min(self.image))
        total_gradients = np.zeros((1, self.image_size, self.image_size, 3),
                                   dtype=np.float32)
        for i in range(self.nsamples):
            image = self.image.copy()
            noise = np.random.normal(0, stddev, self.original_shape)
            image_noise = image + noise

            if self.explainer == "SmoothGrad_IntegratedGradients":
                LOGGER.info('Explanation method %s' % self.explainer)
                saliency, _ = IntegratedGradientExplainer(
                    self.model, self.model_name, image_noise,
                    self.explainer, self.layer_name, self.visualize_index,
                    self.preprocessor_fn, self.image_size, self.steps,
                    self.batch_size, normalize=False,
                    prior_boxes=self.prior_boxes, explaining=self.explaining)
            elif self.explainer == 'SmoothGrad_GuidedBackpropagation':
                LOGGER.info('Explanation method %s' % self.explainer)
                saliency, _ = GuidedBackpropagationExplainer(
                    self.model, self.model_name, image_noise,
                    self.explainer, self.layer_name, self.visualize_index,
                    self.preprocessor_fn, self.image_size, normalize=False,
                    prior_boxes=self.prior_boxes, explaining=self.explaining)
            else:
                raise ValueError("Explanation method not implemented %s"
                                 % self.explainer)

            if self.magnitude:
                total_gradients += (saliency * saliency)
            else:
                total_gradients += saliency

        return total_gradients / self.nsamples


def SmoothGradExplainer(model, model_name, image_path, interpretation_method,
                        layer_name, visualize_index, preprocessor_fn,
                        image_size, standard_deviation=0.15, nsamples=5,
                        magnitude=True, steps=10, batch_size=1,
                        prior_boxes=None, explaining=None):
    sg = SmoothGrad(model, model_name, image_path,
                    interpretation_method,
                    layer_name, visualize_index, preprocessor_fn,
                    image_size, standard_deviation, nsamples,
                    magnitude, steps, batch_size, prior_boxes, explaining)
    saliency = sg.get_saliency_map()
    saliency_stat = (np.min(saliency), np.max(saliency))
    saliency = visualize_saliency_grayscale(saliency)
    return saliency, saliency_stat
