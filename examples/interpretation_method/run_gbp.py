from dext.postprocessing.saliency_visualization import plot_single_saliency
from paz.models.detection.ssd512 import SSD512
from paz.processors.image import LoadImage
from dext.model.ssd.utils import ssd_preprocess
from dext.model.ssd.ssd_postprocess import ssd_postprocess
from paz.datasets.utils import get_class_names
from dext.interpretation_method.guided_backpropagation import (
    GuidedBackpropagationExplainer)

# Get inputs
raw_image_path = "/media/deepan/externaldrive1/project_repos/DEXT_versions/dext/data/000000162701.jpg"
loader = LoadImage()
raw_image = loader(raw_image_path)
image = raw_image.copy()
preprocessing = ssd_preprocess
postprocessing = ssd_postprocess

# Build model
class_names = get_class_names("COCO")
num_classes = len(class_names)
model = SSD512(num_classes=num_classes, weights="COCO")
model.summary()

# Forward pass
input_image, image_scales = preprocessing(image, model.input_shape[1:3])
outputs = model(input_image)
detection_image, detections, class_map_idx = postprocessing(
    model, outputs, image_scales, raw_image.copy())
print(detections)

# Select target neuron to visualize
explain_object = 0
visualize_idx = (0,
                 int(class_map_idx[explain_object][0]),
                 int(class_map_idx[explain_object][1]) + 4)

# Generate explanations
saliency, saliency_stat = GuidedBackpropagationExplainer(
    model, "SSD", raw_image_path, "GBP", "boxes", visualize_idx, preprocessing,
    512)

# Visualize
fig = plot_single_saliency(detection_image, image, saliency,
                           class_map_idx[explain_object][2],
                           class_names[class_map_idx[explain_object][1]],
                           explaining='Classification',
                           interpretation_method='GuidedBackpropagation',
                           model_name='SSD', saliency_stat=saliency_stat)
fig.savefig("ssd512_saliency.jpg")
