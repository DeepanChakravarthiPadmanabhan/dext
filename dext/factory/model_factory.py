from dext.model.efficientdet.efficientdet import EFFICIENTDETD0, EFFICIENTDETD1
from dext.model.efficientdet.efficientdet import EFFICIENTDETD2, EFFICIENTDETD3
from dext.model.efficientdet.efficientdet import EFFICIENTDETD4, EFFICIENTDETD5
from dext.model.efficientdet.efficientdet import EFFICIENTDETD6, EFFICIENTDETD7
from dext.model.efficientdet.efficientdet import EFFICIENTDETD7x
from dext.model.ssd.ssd300_no_pool_model import SSD300NoPool
from paz.models.detection.ssd512 import SSD512
from paz.models.detection.ssd300 import SSD300
from dext.model.faster_rcnn.faster_rcnn_detection_model import (
    faster_rcnn_detection)
from dext.model.marine_debris_ssd.utils import (marine_debris_ssd_vgg16,
                                                marine_debris_ssd_vgg16_random)
from dext.model.marine_debris_ssd_resnet20.utils import (
    marine_debris_ssd_resnet20, marine_debris_ssd_resnet20_random)
from dext.model.marine_debris_ssd_mobilenet.utils import (
    marine_debris_ssd_mobilenet)
from dext.model.marine_debris_ssd_densenet121.utils import (
    marine_debris_ssd_densenet121)
from dext.model.marine_debris_ssd_squeezenet.utils import (
    marine_debris_ssd_squeezenet)
from dext.model.marine_debris_ssd_minixception.utils import (
    marine_debris_ssd_minixception)
from dext.model.marine_debris_ssd_autoencoder.utils import (
    marine_debris_ssd_autoencoder)

class ModelFactory:
    def __init__(self, model_name):
        self.model_name = model_name

    def factory(self):
        if self.model_name == "EFFICIENTDETD0":
            return EFFICIENTDETD0()
        elif self.model_name == "EFFICIENTDETD1":
            return EFFICIENTDETD1()
        elif self.model_name == "EFFICIENTDETD2":
            return EFFICIENTDETD2()
        elif self.model_name == "EFFICIENTDETD3":
            return EFFICIENTDETD3()
        elif self.model_name == "EFFICIENTDETD4":
            return EFFICIENTDETD4()
        elif self.model_name == "EFFICIENTDETD5":
            return EFFICIENTDETD5()
        elif self.model_name == "EFFICIENTDETD6":
            return EFFICIENTDETD6()
        elif self.model_name == "EFFICIENTDETD7":
            return EFFICIENTDETD7()
        elif self.model_name == "EFFICIENTDETD7x":
            return EFFICIENTDETD7x()
        elif self.model_name == "SSD512":
            return SSD512()
        elif self.model_name == 'SSD300':
            return SSD300()
        elif self.model_name == 'SSD300NoPool':
            return SSD300NoPool()
        elif self.model_name == 'FasterRCNN':
            model = faster_rcnn_detection()
            return model
        elif self.model_name == 'MarineDebris_SSD_VGG16':
            model = marine_debris_ssd_vgg16()
            return model
        elif self.model_name == 'MarineDebris_SSD_ResNet20':
            model = marine_debris_ssd_resnet20()
            return model
        elif self.model_name == 'MarineDebris_SSD_MobileNet':
            model = marine_debris_ssd_mobilenet()
            return model
        elif self.model_name == 'MarineDebris_SSD_DenseNet121':
            model = marine_debris_ssd_densenet121()
            return model
        elif self.model_name == 'MarineDebris_SSD_SqueezeNet':
            model = marine_debris_ssd_squeezenet()
            return model
        elif self.model_name == 'MarineDebris_SSD_MiniXception':
            model = marine_debris_ssd_minixception()
            return model
        elif self.model_name == 'MarineDebris_SSD_Autoencoder':
            model = marine_debris_ssd_autoencoder()
            return model
        elif self.model_name == 'MarineDebris_SSD_VGG16_Random':
            model = marine_debris_ssd_vgg16_random()
            return model
        elif self.model_name == 'MarineDebris_SSD_ResNet20_Random':
            model = marine_debris_ssd_resnet20_random()
            return model
        else:
            raise ValueError("Model not implemented %s" % self.model_name)
