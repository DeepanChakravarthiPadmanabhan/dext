from dext.inference.inference import inference_image_ssd
from dext.inference.inference import inference_image_efficientdet
from dext.inference.inference import inference_image_faster_rcnn
from dext.inference.inference import inference_image_marine_debris_ssd_resnet20
from dext.inference.inference import (
    inference_image_marine_debris_ssd_mobilenet)


class InferenceFactory:
    def __init__(self, model_name):
        self.model_name = model_name

    def factory(self):
        if "EFFICIENTDET" in self.model_name:
            return inference_image_efficientdet
        elif self.model_name in ["SSD512", "SSD300"]:
            return inference_image_ssd
        elif "FasterRCNN" in self.model_name:
            return inference_image_faster_rcnn
        elif self.model_name == "MarineDebris_SSD_VGG16":
            return inference_image_ssd
        elif self.model_name == "MarineDebris_SSD_ResNet20":
            return inference_image_marine_debris_ssd_resnet20
        elif self.model_name == "MarineDebris_SSD_MobileNet":
            return inference_image_marine_debris_ssd_mobilenet
        else:
            raise ValueError(
                "Preprocessor not implemented %s" % self.model_name)
