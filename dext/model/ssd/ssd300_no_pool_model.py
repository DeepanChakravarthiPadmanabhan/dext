from paz.models.detection import SSD300NoPool
from paz.datasets.utils import get_class_names

class_names = get_class_names("COCO")
num_classes = len(class_names)

model = SSD300NoPool(num_classes=num_classes, base_weights="VOC",
                     head_weights=None)
model.summary()

# TODO: Preprocess image for ssd
# TODO: Postprocess image for ssd
# TODO: Select neuron in the model output corresponding to the box class
#  and offset
# TODO: Do BP

# TODO: Finally integrate the procedure with explain_model method in explainer
