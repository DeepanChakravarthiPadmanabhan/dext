# DExT: Detector Explanation Toolkit

The objective of the repository is to provide a toolkit to explain the detections of an object detector using the complete bounding box and class predictions.

As object detectors are prone to non-local effects and utilize context, providing the rationale behind each components of the detection, the complete detection, and all detections in an image is essential.

A poster with preliminary results can be found [here](https://github.com/DeepanChakravarthiPadmanabhan/mtdocuments/blob/master/PadmanabhanDC_MTPoster.pdf).

This is part of the thesis work supervised by [Dr. Matias Valdenegro-Toro](https://mvaldenegro.github.io/) and [Professor. Dr. Paul G Plöger](https://www.h-brs.de/en/inf/prof-dr-paul-g-ploeger).
The thesis proposal can be found [here](https://github.com/DeepanChakravarthiPadmanabhan/mtdocuments/blob/master/PadmanabhanDC-MTProposal/PadmanabhanDC-MTProposal.pdf).


## Installation
```
pip install -e .
```

## Generate explanations for a single image
```
dext_explainer -c config/explainer.gin -i <interpretation_method> -m <model_name> --explain_mode single_image --input_image_path <image_path>
```

## Generate explanations on COCO dataset images
```
dext_explainer -c config/explainer.gin -i <interpretation_method> -m <model_name> --num_images <number_of_images>
```

## Evaluate explanations
```
dext_evaluator -c config/evaluator.gin -i GuidedBackpropagation -m SSD512
```

## Important explainer arguments
| Argument                 | Description                                                    |
|--------------------------|----------------------------------------------------------------|
| config, c                | configuration file with dataset paths                          |
| interpretation_method, i | name of the explanation method                                 |
| model_name, m            | name of the model to explain                                   |
| explain_mode             | single image explanation or explain random images from dataset |
| dataset_name             | name of the dataset                                            |


## Object detectors available
1. EFFICIENTDETD{x}, x = 0, 1, 2, 3, 4, 5, 6, 7, 7x
2. SSD512
3. FasterRCNN

All above detectors are available with COCO weights. The weights are ported from [1], [2], [3]. 

## Interpretation methods available
1. IntergratedGradients (IG)
2. GuidedBackpropagation (GBP)
3. SmoothGrad_Integrated Gradients (SIG)
4. SmoothGrad_GuidedBackpropagation (SGBP)
5. GradCAM, Gradient-weighted Class Activation Mapping (GradCAM)
6. LIME, Local Interpretable Model-agnostic Explanations (LIME)
7. DeepExplainer, SHapley Additive exPlanations (SHAP) + DeepLIFT
8. GradientExplainer, SHapley Additive exPlanations (SHAP) + Integrated Gradients (IG) + SmoothGrad (SG)

## References

[1] Google AutoML, "EfficientDet", GitHub, Available at: https://github.com/google/automl/tree/master/efficientdet, Accessed on: 01. 06. 2021.

[2] Octavio Arriaga, Matias Valdenegro-Toro, Mohandass Muthuraja, Sushma Devaramani, and Frank Kirchner. "Perception for Autonomous Systems (PAZ)." arXiv preprint arXiv:2010.14541. 2020.

[3] Waleed Abdulla, "Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow", GitHub, Available at: https://github.com/matterport/Mask_RCNN, Accessed on: 01.06. 2021.

[4] Maximilian Alber, Sebastian Lapuschkin, Philipp Seegerer, Miriam Hägele, Kristof T. Schütt, Grégoire Montavon, Wojciech Samek, Klaus-Robert Müller, Sven Dähne, and Pieter-Jan Kindermans. "iNNvestigate neural networks!." Journal of Machine Learning Research 20, No. 93, pp. 1-8. 2019.

[5] Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. ""Why should I trust you?" Explaining the predictions of any classifier." In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining, pp. 1135-1144. 2016.

[6] Scott M. Lundberg and Su-In Lee. "A unified approach to interpreting model predictions." In Proceedings of the 31st international conference on neural information processing systems, pp. 4768-4777. 2017.

[7] François Chollet, "Visualizing what convnets learn", Keras documentation, Available at: https://keras.io/examples/vision/visualizing_what_convnets_learn/, Accessed on: 03. 05. 2021.

[8] Jacob Gildenblat, "Visualizing CNN filters with keras", GitHub, Available at: https://jacobgil.github.io/deeplearning/filter-visualizations, Accessed on: 03. 05. 2021.

[9] Vincent Kao, "Visualize output of CNN filter by gradient ascent", Kaggle, Available at: https://www.kaggle.com/vincentman0403/visualize-output-of-cnn-filter-by-gradient-ascent, Accessed on: 03. 05. 2021.

[10] Sicara, "tf-explain", GitHub, Available at: https://github.com/sicara/tf-explain, Accessed on: 03. 05. 2021.

[11] Cristian Vasta, Deep Learning Models with Tensorflow 2.0, Available at: https://morioh.com/p/64064daff26c, Accessed on: 03. 05. 2021.

[12] Hoa Nguyen, "CNN Visualization Keras TF2", GitHub, Available at: https://github.com/nguyenhoa93/cnn-visualization-keras-tf2, Accessed on: 03. 05. 2021.

[13] People + AI Research, Google Research, "Saliency Methods", GitHub, Available at: https://github.com/PAIR-code/saliency, Accessed on: 01. 06. 2021.

[14] Marco Ancona, "DeepExplain: attribution methods for Deep Learning", GitHub, Available at: https://github.com/marcoancona/DeepExplain, Accessed on: 01. 06. 2021.

[15] Maximilian Alber, "iNNvestigate neural networks!", GitHub, Available at: https://github.com/albermax/innvestigate, Accessed on: 03. 05. 2021.

[16] Marco Tulio Correia, "lime", GitHub, Available at: https://github.com/marcotcr/lime, Accessed on: 01. 06. 2021.

[17] Scott M. Lundberg, "shap", GitHub, Available at: https://github.com/slundberg/shap, Accessed on: 01. 06. 2021.

[18] Mingxing Tan, Ruoming Pang, and Quoc V. Le. "Efficientdet: Scalable and efficient object detection." In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10781-10790. 2020.

[19] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, and Alexander C. Berg. "Ssd: Single shot multibox detector." In European conference on computer vision, pp. 21-37. Springer, Cham, 2016.