class Explainer(object):

    def __init__(self, model, model_name, image, explainer):
        self.model = model
        self.model_name = model_name
        self.image = image
        self.explainer = explainer

    def get_saliency_map(self):
        raise NotImplementedError('Interpretation method not implemented.')

    def build_custom_model(self):
        raise NotImplementedError('Custom model is not built.')

    @property
    def explainer(self):
        return self._explainer

    @explainer.setter
    def explainer(self, explainer):
        self._explainer = explainer
