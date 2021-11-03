class Explainer(object):

    def __init__(self, model_name, image, explainer):
        pass

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
