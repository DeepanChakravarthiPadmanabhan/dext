from dext.model.efficientdet.utils import efficientdet_preprocess
from dext.model.ssd.utils import ssd_preprocess
from dext.model.faster_rcnn.faster_rcnn_preprocess import (
    faster_rcnn_preprocess)
from dext.model.marine_debris_ssd.utils import (
    marine_debris_ssd_vgg16_preprocess)
from dext.model.marine_debris_ssd_mobilenet.utils import (
    marine_debris_ssd_mobilenet_preprocess)
from dext.model.marine_debris_ssd_resnet20.utils import (
    marine_debris_ssd_resnet20_preprocess)


class PreprocessorFactory:
    def __init__(self, model_name):
        self.model_name = model_name

    def factory(self):
        if "EFFICIENTDET" in self.model_name:
            return efficientdet_preprocess
        elif self.model_name in ["SSD512", "SSD300"]:
            return ssd_preprocess
        elif "FasterRCNN" in self.model_name:
            return faster_rcnn_preprocess
        elif "MarineDebris_SSD_VGG16" == self.model_name:
            return marine_debris_ssd_vgg16_preprocess
        elif "MarineDebris_SSD_ResNet20" == self.model_name:
            return marine_debris_ssd_resnet20_preprocess
        elif "MarineDebris_SSD_MobileNet" == self.model_name:
            return marine_debris_ssd_mobilenet_preprocess
        else:
            raise ValueError(
                "Preprocessor not implemented %s" % self.model_name)
