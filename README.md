# DExT: Detector Explanation Toolkit

## Installation
```
pip install -e .
```

## Object detectors with weights available
1. EfficientDet - COCO
2. SSD512 - COCO

## Interpretation methods available
1. Intergrated Gradients (IG)
2. Guided Backpropagation (GBP)
3. SmoothGrad (SG) with Integrated Gradients (SIG)
4. SmoothGrad (SG) with Guided Backpropagation (SGBP)
5. Gradient-weighted Class Activation Mapping (GradCAM)
6. Local Interpretable Model-agnostic Explanations (LIME)
7. DeepExplainer (DE) - SHapley Additive exPlanations (SHAP) + DeepLIFT
8. GradientExplainer (GE) - SHapley Additive exPlanations (SHAP) + Integrated Gradients (IG) + SmoothGrad (SG)

## References
[1] Octavio Arriaga, Matias Valdenegro-Toro, Mohandass Muthuraja, Sushma Devaramani, and Frank Kirchner. "Perception for Autonomous Systems (PAZ)." arXiv preprint arXiv:2010.14541. 2020.

[2] https://keras.io/examples/vision/visualizing_what_convnets_learn/

[3] https://jacobgil.github.io/deeplearning/filter-visualizations

[4] https://www.kaggle.com/vincentman0403/visualize-output-of-cnn-filter-by-gradient-ascent

[5] https://medium.com/@jon.froiland/convolutional-neural-networks-part-8-3ac54c9478cc

[6] Sicara, "tf-explain", Available at: https://github.com/sicara/tf-explain#available-methods, Accessed on: 03. 05. 2021.

[7] Cristian Vasta, Deep Learning Models with Tensorflow 2.0, Available at: https://morioh.com/p/64064daff26c, Accessed on: 03. 05. 2021.

[8] Hoa Nguyen, CNN Visualization Keras TF2, GitHub, Available at: https://github.com/nguyenhoa93/cnn-visualization-keras-tf2, Accessed on: 03. 05. 2021.

[9] People + AI Research, Google Research, "Saliency Methods", Available at: https://github.com/PAIR-code/saliency, Accessed on: 01. 06. 2021.

[10] Marco Ancona, "DeepExplain", Available at: https://github.com/marcoancona/DeepExplain, Accessed on: 01. 06. 2021.

[11] https://github.com/albermax/innvestigate

[12] Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. "" Why should i trust you?" Explaining the predictions of any classifier." In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining, pp. 1135-1144. 2016.

[13] Marco Tulio Correia, "lime", Available at: https://github.com/marcotcr/lime, Accessed on: 01. 06. 2021.

[14] Scott M. Lundberg and Su-In Lee. "A unified approach to interpreting model predictions." In Proceedings of the 31st international conference on neural information processing systems, pp. 4768-4777. 2017.

[15] Scott M. Lundberg, "shap", Available at: https://github.com/slundberg/shap, Accessed on: 01. 06. 2021.