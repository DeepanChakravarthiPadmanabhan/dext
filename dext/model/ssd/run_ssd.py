from paz.models.detection.ssd512 import SSD512
from paz.datasets.utils import get_class_names
from paz.processors.image import LoadImage
import matplotlib.pyplot as plt
from dext.interpretation_method.guided_backpropagation import \
    GuidedBackpropagationExplainer
from dext.postprocessing.saliency_visualization import \
    visualize_saliency_grayscale
from dext.model.ssd.utils import ssd_preprocess
from dext.model.ssd.ssd_postprocess import ssd_postprocess
from dext.postprocessing.saliency_visualization import plot_single_saliency
from dext.model.utils import get_all_layers

class_names = get_class_names("COCO")
num_classes = len(class_names)
model = SSD512(num_classes=num_classes, weights="COCO")
model.summary()
all_layers = get_all_layers(model)
raw_image = "images/000000309391.jpg"
loader = LoadImage()
raw_image = loader(raw_image)
image = raw_image.copy()
preprocessing = ssd_preprocess
postprocessing = ssd_postprocess
input_image, image_scales = preprocessing(image, model.input_shape[1:3])
outputs = model(input_image)
detection_image, detections, class_map_idx = postprocessing(
    model, outputs, image_scales, raw_image.copy())
plt.imsave("ssd512.jpg", detection_image)
print(detections)
explain_object = 0
visualize_idx = (0,
                 int(class_map_idx[explain_object][0]),
                 int(class_map_idx[explain_object][1]) + 4)
print("class map idx: ", class_map_idx)
print('visualizer: ', visualize_idx)
saliency = GuidedBackpropagationExplainer(
    model, "SSD", raw_image, "boxes", visualize_idx, preprocessing, 512)
saliency = visualize_saliency_grayscale(saliency)
fig = plot_single_saliency(detection_image, image, saliency,
                           class_map_idx[explain_object][2],
                           class_names[class_map_idx[explain_object][1]],
                           explaining='Classification',
                           interpretation_method='GuidedBackpropagation',
                           model_name='SSD')
fig.savefig("ssd512_saliency.jpg")
