from dext.interpretation_method.integrated_gradient import \
    IntegratedGradientExplainer
from dext.interpretation_method.guided_backpropagation import \
    GuidedBackpropagationExplainer
from dext.interpretation_method.grad_cam import GradCAMExplainer


class ExplainerFactory:
    def __init__(self, explainer):
        self.explainer = explainer

    def factory(self):
        if self.explainer == "IntegratedGradients":
            return IntegratedGradientExplainer
        elif self.explainer == "GuidedBackpropagation":
            return GuidedBackpropagationExplainer
        elif self.explainer == "GradCAM":
            return GradCAMExplainer
        else:
            raise ValueError(
                "Architecture name not implemented %s" % (self.explainer))
