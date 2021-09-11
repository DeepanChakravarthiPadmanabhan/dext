from paz.models.detection.ssd300 import SSD300
from paz.datasets.utils import get_class_names

class_names = get_class_names("COCO")
num_classes = len(class_names)

model = SSD300(base_weights="VOC", head_weights=None, num_classes=num_classes)
model.load_weights("weights.06-8.61.hdf5")
model.summary()


# TODO: Preprocess image for ssd
# TODO: Postprocess image for ssd
# TODO: Select neuron in the model output corresponding to the box class
#  and offset
# TODO: Do BP

# TODO: Finally integrate the procedure with explain_model method in explainer