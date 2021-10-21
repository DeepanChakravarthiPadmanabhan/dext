# DExT: Detector Explanation Toolkit

## Installation
```
pip install -e .
```

## Generate explanations
```
dext_explainer -c config/explainer.gin -i GuidedBackpropagation -m SSD512
```

## Object detectors with weights available
1. EfficientDet - COCO
2. SSD512 - COCO
3. FasterRCNN - COCO

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

[2] Maximilian Alber, Sebastian Lapuschkin, Philipp Seegerer, Miriam Hägele, Kristof T. Schütt, Grégoire Montavon, Wojciech Samek, Klaus-Robert Müller, Sven Dähne, and Pieter-Jan Kindermans. "iNNvestigate neural networks!." Journal of Machine Learning Research 20, No. 93, pp. 1-8. 2019.

[3] Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. ""Why should I trust you?" Explaining the predictions of any classifier." In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining, pp. 1135-1144. 2016.

[4] Scott M. Lundberg and Su-In Lee. "A unified approach to interpreting model predictions." In Proceedings of the 31st international conference on neural information processing systems, pp. 4768-4777. 2017.

[5] François Chollet, "Visualizing what convnets learn", Keras documentation, Available at: https://keras.io/examples/vision/visualizing_what_convnets_learn/, Accessed on: 03. 05. 2021.

[6] Jacob Gildenblat, "Visualizing CNN filters with keras", GitHub, Available at: https://jacobgil.github.io/deeplearning/filter-visualizations, Accessed on: 03. 05. 2021.

[7] Vincent Kao, "Visualize output of CNN filter by gradient ascent", Kaggle, Available at: https://www.kaggle.com/vincentman0403/visualize-output-of-cnn-filter-by-gradient-ascent, Accessed on: 03. 05. 2021.

[8] Sicara, "tf-explain", GitHub, Available at: https://github.com/sicara/tf-explain, Accessed on: 03. 05. 2021.

[9] Cristian Vasta, Deep Learning Models with Tensorflow 2.0, Available at: https://morioh.com/p/64064daff26c, Accessed on: 03. 05. 2021.

[10] Hoa Nguyen, "CNN Visualization Keras TF2", GitHub, Available at: https://github.com/nguyenhoa93/cnn-visualization-keras-tf2, Accessed on: 03. 05. 2021.

[11] People + AI Research, Google Research, "Saliency Methods", GitHub, Available at: https://github.com/PAIR-code/saliency, Accessed on: 01. 06. 2021.

[12] Marco Ancona, "DeepExplain: attribution methods for Deep Learning", GitHub, Available at: https://github.com/marcoancona/DeepExplain, Accessed on: 01. 06. 2021.

[13] Maximilian Alber, "iNNvestigate neural networks!", GitHub, Available at: https://github.com/albermax/innvestigate, Accessed on: 03. 05. 2021.

[14] Marco Tulio Correia, "lime", GitHub, Available at: https://github.com/marcotcr/lime, Accessed on: 01. 06. 2021.

[15] Scott M. Lundberg, "shap", GitHub, Available at: https://github.com/slundberg/shap, Accessed on: 01. 06. 2021.