from dext.interpretation_method.integrated_gradient import (
    IntegratedGradientExplainer)
from dext.interpretation_method.guided_backpropagation import (
    GuidedBackpropagationExplainer)
from dext.interpretation_method.grad_cam import GradCAMExplainer
from dext.interpretation_method.smooth_grad import SmoothGradExplainer
from dext.interpretation_method.lime_explainer import LimeExplainer
from dext.interpretation_method.shap_deep_explainer import SHAP_DeepExplainer
from dext.interpretation_method.shap_gradient_explainer import (
    SHAP_GradientExplainer)
# from dext.interpretation_method.relevance_propagation import OD_LRPExplainer


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
        elif self.explainer == "SHAP_DeepExplainer":
            return SHAP_DeepExplainer
        elif self.explainer == "SHAP_GradientExplainer":
            return SHAP_GradientExplainer
        # elif self.explainer == "LRP":
        #     return OD_LRPExplainer
        else:
            raise ValueError("Explanation method not implemented %s"
                             % self.explainer)
