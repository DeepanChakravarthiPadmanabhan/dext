from dext.model.efficientdet.efficientdet_postprocess import (
    efficientdet_postprocess)
from dext.model.ssd.ssd_postprocess import ssd_postprocess
from dext.model.faster_rcnn.faster_rcnn_postprocess import (
    faster_rcnn_postprocess)
from dext.model.marine_debris_ssd.utils import (
    marine_debris_ssd_vgg16_postprocess)
from dext.model.marine_debris_ssd_resnet20.utils import (
    marine_debris_ssd_resnet20_postprocess)
from dext.model.marine_debris_ssd_mobilenet.utils import (
    marine_debris_ssd_mobilenet_postprocess)
from dext.model.marine_debris_ssd_densenet121.utils import (
    marine_debris_ssd_densenet121_postprocess)
from dext.model.marine_debris_ssd_squeezenet.utils import (
    marine_debris_ssd_squeezenet_postprocess)
from dext.model.marine_debris_ssd_minixception.utils import (
    marine_debris_ssd_minixception_postprocess)
from dext.model.marine_debris_ssd_autoencoder.utils import (
    marine_debris_ssd_autoencoder_postprocess)


class PostprocessorFactory:
    def __init__(self, model_name):
        self.model_name = model_name

    def factory(self):
        if "EFFICIENTDET" in self.model_name:
            return efficientdet_postprocess
        elif self.model_name in ["SSD512", "SSD300"]:
            return ssd_postprocess
        elif "FasterRCNN" in self.model_name:
            return faster_rcnn_postprocess
        elif "MarineDebris_SSD_VGG16" == self.model_name:
            return marine_debris_ssd_vgg16_postprocess
        elif "MarineDebris_SSD_ResNet20" == self.model_name:
            return marine_debris_ssd_resnet20_postprocess
        elif "MarineDebris_SSD_MobileNet" == self.model_name:
            return marine_debris_ssd_mobilenet_postprocess
        elif "MarineDebris_SSD_DenseNet121" == self.model_name:
            return marine_debris_ssd_densenet121_postprocess
        elif "MarineDebris_SSD_SqueezeNet" == self.model_name:
            return marine_debris_ssd_squeezenet_postprocess
        elif "MarineDebris_SSD_MiniXception" == self.model_name:
            return marine_debris_ssd_minixception_postprocess
        elif "MarineDebris_SSD_Autoencoder" == self.model_name:
            return marine_debris_ssd_autoencoder_postprocess
        else:
            raise ValueError(
                "Preprocessor not implemented %s" % self.model_name)
