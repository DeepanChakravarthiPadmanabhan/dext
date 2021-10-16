import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from paz.processors.image import LoadImage
import cv2
raw_image = "images/000000128224.jpg"
loader = LoadImage()
image = loader(raw_image)
image = cv2.resize(image, (640, 640))
image = np.asarray(image).astype('uint8')
image = image[np.newaxis]
print(type(image), image.shape)
module_path = 'https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1'
detector = hub.load(module_path)
detector_output = detector(image)
class_ids = detector_output["detection_classes"]
boxes = detector_output['detection_boxes']
input_layer = tf.keras.Input(shape=(640, 640, 3), name="image")
out_layer = hub.KerasLayer(module_path, trainable=False)
layer_out = out_layer(image)
print(layer_out.keys())
for i in layer_out.keys():
    print(layer_out[i].shape)
