from dext.inference.inference import inference_image_ssd
from dext.inference.inference import inference_image_efficientdet
from dext.inference.inference import inference_image_faster_rcnn
from dext.inference.inference import inference_image_marine_debris_ssd_resnet20
from dext.inference.inference import (
    inference_image_marine_debris_ssd_mobilenet)
from dext.inference.inference import (
    inference_image_marine_debris_ssd_densenet121)
from dext.inference.inference import (
    inference_image_marine_debris_ssd_squeezenet)
from dext.inference.inference import (
    inference_image_marine_debris_ssd_minixception)
from dext.inference.inference import (
    inference_image_marine_debris_ssd_autoencoder)


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
        elif self.model_name == "MarineDebris_SSD_ResNet20_Random":
            return inference_image_marine_debris_ssd_resnet20
        elif self.model_name == "MarineDebris_SSD_MobileNet":
            return inference_image_marine_debris_ssd_mobilenet
        elif self.model_name == "MarineDebris_SSD_DenseNet121":
            return inference_image_marine_debris_ssd_densenet121
        elif self.model_name == "MarineDebris_SSD_SqueezeNet":
            return inference_image_marine_debris_ssd_squeezenet
        elif self.model_name == "MarineDebris_SSD_MiniXception":
            return inference_image_marine_debris_ssd_minixception
        elif self.model_name == "MarineDebris_SSD_Autoencoder":
            return inference_image_marine_debris_ssd_autoencoder
        else:
            raise ValueError(
                "Preprocessor not implemented %s" % self.model_name)
