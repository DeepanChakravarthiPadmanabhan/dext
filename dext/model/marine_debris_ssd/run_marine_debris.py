import cv2
import numpy as np
import tensorflow as tf
from paz.models.detection.ssd300 import SSD300
from dext.utils.class_names import get_classes
import matplotlib.pyplot as plt
from dext.interpretation_method.guided_backpropagation import \
    GuidedBackpropagationExplainer
from dext.model.marine_debris_ssd.utils import marine_debris_ssd_preprocess
from dext.model.marine_debris_ssd.utils import marine_debris_ssd_postprocess
from dext.postprocessing.saliency_visualization import plot_single_saliency
from dext.model.utils import get_all_layers

analyzing_label = 'correct'

class_names = get_classes("MarineDebris", "SSD300")
num_classes = len(class_names)
model = SSD300(12, None, None)
model.load_weights('/media/deepan/externaldrive1/project_repos/DEXT_versions/weights/weights_vgg16_random.hdf5')
model.summary()

all_layers = get_all_layers(model)

raw_image_path = "images/marine-debris-aris3k-162.png"
raw_image = cv2.imread(raw_image_path, cv2.IMREAD_GRAYSCALE)
raw_image = np.stack((raw_image,) * 3, axis=-1)
image = raw_image.copy()


input_image, image_scales = marine_debris_ssd_preprocess(
    image, model.input_shape[1:3])
outputs = model(input_image)
detection_image, detections, class_map_idx = marine_debris_ssd_postprocess(
    model, outputs, image_scales, raw_image.copy())
plt.imsave("ssd512.jpg", detection_image)
print(detections)

explain_object = 2
# visualize_idx = (0,
#                  int(class_map_idx[explain_object][0]),
#                  int(class_map_idx[explain_object][1]) + 4)
visualize_idx = (0, 4009, 8)

print("class map idx: ", class_map_idx)
print('visualizer: ', visualize_idx)

saliency, saliency_stat = GuidedBackpropagationExplainer(
    model, "SSD", raw_image_path, "GBP", "boxes", visualize_idx,
    marine_debris_ssd_preprocess, 300)

confidence = class_map_idx[explain_object][2]
class_name = class_names[class_map_idx[explain_object][1]]
fig = plot_single_saliency(detection_image, image, saliency, confidence,
                           class_name, explaining='Classification',
                           interpretation_method='GuidedBackpropagation',
                           model_name='SSD', saliency_stat=saliency_stat)

fig.savefig("ssd512_saliency.jpg")

# [[6653.0, 3, 0.9990842342376709], [4009.0, 4, 0.9990542531013489],
 # [622.0, 11, 0.7800893187522888]]
