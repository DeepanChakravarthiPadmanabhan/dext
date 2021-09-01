from dext.interpretation_method.integrated_gradient import IntegratedGradientExplainer
from dext.interpretation_method.guided_backpropagation import GuidedBackpropagationExplainer

class ExplainerFactory:
    def __init__(self, explainer):
        self.explainer = explainer

    def factory(self):
        if self.explainer == "IntegratedGradients":
            return IntegratedGradientExplainer
        elif self.explainer == "GuidedBackpropagation":
            return GuidedBackpropagationExplainer
        else:
            raise ValueError("Architecture name not implemented %s" % (self.explainer))