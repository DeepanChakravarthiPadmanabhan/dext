from dext.interpretation_method.integrated_gradient import \
    IntegratedGradientExplainer
from dext.interpretation_method.guided_backpropagation import \
    GuidedBackpropagationExplainer
from dext.interpretation_method.grad_cam import GradCAMExplainer
from dext.interpretation_method.smooth_grad import SmoothGradExplainer
from dext.interpretation_method.lime_explainer import LimeExplainer


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
        elif "SmoothGrad" in self.explainer:
            return SmoothGradExplainer
        elif self.explainer == "LIME":
            return LimeExplainer
        else:
            raise ValueError("Explanation method not implemented %s"
                             % self.explainer)
