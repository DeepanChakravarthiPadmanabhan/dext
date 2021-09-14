import tensorflow as tf
import matplotlib.pyplot as plt
from dext.model.efficientdet.efficientdet import EFFICIENTDETD0
from paz.processors.image import LoadImage
from dext.model.efficientdet.utils import efficientdet_preprocess
from dext.model.efficientdet.efficientdet_postprocess import \
    efficientdet_postprocess
from dext.model.efficientdet.efficientdet_postprocess import \
    get_class_name_efficientdet
from dext.interpretation_method.integrated_gradient import \
    IntegratedGradientExplainer
from dext.postprocessing.saliency_visualization import plot_single_saliency
from dext.postprocessing.saliency_visualization import \
    visualize_saliency_grayscale
from dext.model.functional_models import get_functional_model
from dext.model.utils import get_all_layers


class_names = get_class_name_efficientdet('COCO')
raw_image = "images/000000128224.jpg"
loader = LoadImage()
raw_image = loader(raw_image)
image = raw_image.copy()
image, image_scales = efficientdet_preprocess(image, 512)
model = EFFICIENTDETD0()
model_fun = get_functional_model("EFFICIENTDETD0", model)
custom_model = tf.keras.Model(
    inputs=[model_fun.inputs],
    outputs=[model_fun.output])
all_layers = get_all_layers(custom_model)
outputs = model(image)
detection_image, detections, class_map_idx = efficientdet_postprocess(
    model, outputs, image_scales, raw_image)
print(detections)
plt.imsave("efficientdet.jpg", detection_image)
explain_object = 1
visualize_idx = (0,
                 int(class_map_idx[explain_object][0]),
                 int(class_map_idx[explain_object][1]) + 4)
print("class map idx: ", class_map_idx)
print('visualizer: ', visualize_idx)
saliency = IntegratedGradientExplainer(model, "EFFICIENTDET", raw_image,
                                       "boxes", visualize_idx,
                                       efficientdet_preprocess, 512)
saliency = visualize_saliency_grayscale(saliency)
fig = plot_single_saliency(detection_image, raw_image, saliency,
                           class_map_idx[explain_object][2],
                           class_names[class_map_idx[explain_object][1]],
                           explaining='Classification',
                           interpretation_method='IG',
                           model_name='EfficientDet')
fig.savefig("efficientdet_saliency.jpg")
