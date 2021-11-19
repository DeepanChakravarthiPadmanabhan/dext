import tensorflow as tf
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
raw_image_path = "images/000000309391.jpg"
loader = LoadImage()
raw_image = loader(raw_image_path)
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

conv = 0
non_conv = 0
total_weights = len(model.weights)
percent_alter = 100
selected_weights = int((total_weights * percent_alter) / 100)
for n, i in enumerate(model.weights[::-1][:selected_weights]):
    new_shape = model.weights[n].shape
    print('Reinitializing weights for layer: ', model.weights[n].name)
    if 'gamma' in model.weights[n].name:
        model.weights[n].assign(
            tf.constant(1, tf.float32, shape=list(new_shape)))
        non_conv = non_conv + 1
    elif 'beta' in model.weights[n].name:
        model.weights[n].assign(
            tf.constant(0, tf.float32, shape=list(new_shape)))
        non_conv = non_conv + 1
    elif 'WSM' in model.weights[n].name:
        model.weights[n].assign(
            tf.constant(1, tf.float32, shape=list(new_shape)))
        non_conv = non_conv + 1
    elif 'mean' in model.weights[n].name:
        model.weights[n].assign(
            tf.constant(0, tf.float32, shape=list(new_shape)))
        non_conv = non_conv + 1
    elif 'variance' in model.weights[n].name:
        model.weights[n].assign(
            tf.constant(1, tf.float32, shape=list(new_shape)))
        non_conv = non_conv + 1
    elif 'bias' in model.weights[n].name:
        model.weights[n].assign(
            tf.constant(0, tf.float32, shape=list(new_shape)))
        non_conv = non_conv + 1
    else:
        model.weights[n].assign(tf.Variable(
            tf.keras.initializers.GlorotUniform()(shape=list(new_shape),
                                                  dtype=tf.float32)))
        conv = conv + 1
print('CONV NONCONV', conv, non_conv, total_weights)

saliency, saliency_stat = GuidedBackpropagationExplainer(
    model, "SSD", raw_image_path, "GBP", "boxes", visualize_idx, preprocessing,
    512)
fig = plot_single_saliency(detection_image, image, saliency,
                           class_map_idx[explain_object][2],
                           class_names[class_map_idx[explain_object][1]],
                           explaining='Classification',
                           interpretation_method='GuidedBackpropagation',
                           model_name='SSD', saliency_stat=saliency_stat)
fig.savefig("ssd512_saliency.jpg")
