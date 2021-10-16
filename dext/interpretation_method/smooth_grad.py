import logging
import numpy as np

from paz.backend.image import resize_image
from dext.model.mask_rcnn.mask_rcnn_preprocess import ResizeImages
from dext.explainer.utils import get_model
from dext.abstract.explanation import Explainer
from dext.interpretation_method.integrated_gradient \
    import IntegratedGradientExplainer
from dext.interpretation_method.guided_backpropagation \
    import GuidedBackpropagationExplainer
from dext.postprocessing.saliency_visualization import \
    visualize_saliency_grayscale
from dext.explainer.utils import get_model

LOGGER = logging.getLogger(__name__)


class SmoothGrad(Explainer):
    def __init__(self, model, model_name, image,
                 explainer="SmoothGrad_IntegreatedGradients",
                 layer_name=None, visualize_index=None,
                 preprocessor_fn=None, image_size=512, standard_deviation=0.15,
                 nsamples=5, magnitude=1, steps=5, batch_size=1):
        """
        Model: pre-softmax layer (logit layer)
        :param model:
        :param layer_name:
        """
        super().__init__(model, model_name, image, explainer)
        self.model = model
        self.model_name = model_name
        self.image = image
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
        self.image = self.check_image_size(self.image, self.image_size)

    def check_image_size(self, image, image_size):
        if image.shape != (image_size, image_size, 3):
            if self.model_name == 'FasterRCNN':
                resizer = ResizeImages(image_size, 0, image_size, "square")
                image = resizer(image)[0]
            else:
                image = resize_image(image, (image_size, image_size))
        return image

    def get_saliency_map(self):
        stddev = self.standard_deviation * (np.max(self.image) -
                                            np.min(self.image))
        total_gradients = np.zeros((1, self.image_size, self.image_size, 3),
                                   dtype=np.float32)
        for i in range(self.nsamples):
            image = self.image.copy()
            noise = np.random.normal(0, stddev, self.image.shape)
            image_noise = image + noise

            if self.explainer == "SmoothGrad_IntegratedGradients":
                LOGGER.info('Explanation method %s' % self.explainer)
                saliency = IntegratedGradientExplainer(
                    get_model(self.model_name), self.model_name, image_noise,
                    self.explainer, self.layer_name, self.visualize_index,
                    self.preprocessor_fn, self.image_size, self.steps,
                    self.batch_size)
            elif self.explainer == 'SmoothGrad_GuidedBackpropagation':
                LOGGER.info('Explanation method %s' % self.explainer)
                saliency = GuidedBackpropagationExplainer(
                    get_model(self.model_name), self.model_name, image_noise,
                    self.explainer, self.layer_name, self.visualize_index,
                    self.preprocessor_fn, self.image_size)
            else:
                raise ValueError("Explanation method not implemented %s"
                                 % self.explainer)

            if self.magnitude:
                total_gradients += (saliency * saliency)
            else:
                total_gradients += saliency

        return total_gradients / self.nsamples


def SmoothGradExplainer(model_name, image, interpretation_method,
                        layer_name, visualize_index,
                        preprocessor_fn, image_size,
                        standard_deviation=0.15,
                        nsamples=1, magnitude=True, steps=5, batch_size=1):
    model = get_model(model_name, image, image_size)
    sg = SmoothGrad(model, model_name, image,
                    interpretation_method,
                    layer_name, visualize_index, preprocessor_fn,
                    image_size, standard_deviation, nsamples,
                    magnitude, steps, batch_size)
    saliency = sg.get_saliency_map()
    saliency = visualize_saliency_grayscale(saliency)
    return saliency
