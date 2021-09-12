import matplotlib.pyplot as plt
from dext.model.efficientdet.efficientdet import EFFICIENTDETD0
from paz.processors.image import LoadImage
from dext.model.efficientdet.utils import efficientdet_preprocess
from dext.model.efficientdet.efficientdet_postprocess import efficientdet_postprocess
from dext.model.efficientdet.efficientdet_postprocess import get_class_name_efficientdet
from dext.interpretation_method.integrated_gradient import IntegratedGradientExplainer
class_names = get_class_name_efficientdet('COCO')
raw_image = "images/surfboard.jpg"
loader = LoadImage()
raw_image = loader(raw_image)
image = raw_image.copy()
image, image_scales = efficientdet_preprocess(image, 512)
model = EFFICIENTDETD0()
outputs = model(image)
print(outputs.shape)
detection_image, detections, class_map_idx = efficientdet_postprocess(
    model, outputs, image_scales, raw_image)
print(detections)
plt.imsave("efficientdet.jpg", detection_image)

explain_object = 0
visualize_idx = (0,
                 int(class_map_idx[explain_object][0]),
                 int(class_map_idx[explain_object][1]) + 4)
print("class map idx: ", class_map_idx)
print('visualizer: ', visualize_idx)
saliency = IntegratedGradientExplainer(model, "EFFICIENTDET", raw_image,
                                "boxes", visualize_idx,
                                efficientdet_preprocess, 512)

from dext.postprocessing.saliency_visualization import plot_single_saliency
from dext.postprocessing.saliency_visualization import \
    visualize_saliency_grayscale
saliency = visualize_saliency_grayscale(saliency)
fig = plot_single_saliency(detection_image, raw_image, saliency,
                     confidence=class_map_idx[explain_object][2],
                     class_name=class_names[class_map_idx[explain_object][1]],
                     explaining='Classification',
                     interpretation_method='IG',
                     model_name='EfficientDet')
fig.savefig("efficientdet_saliency.jpg")
