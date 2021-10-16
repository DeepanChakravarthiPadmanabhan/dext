from dext.model.efficientdet.efficientdet import EFFICIENTDETD0, EFFICIENTDETD1
from dext.model.efficientdet.efficientdet import EFFICIENTDETD2, EFFICIENTDETD3
from dext.model.efficientdet.efficientdet import EFFICIENTDETD4, EFFICIENTDETD5
from dext.model.efficientdet.efficientdet import EFFICIENTDETD6, EFFICIENTDETD7
from dext.model.efficientdet.efficientdet import EFFICIENTDETD7x
from dext.model.ssd.ssd300_no_pool_model import SSD300NoPool
from paz.models.detection.ssd512 import SSD512
from paz.models.detection.ssd300 import SSD300
from dext.model.mask_rcnn.mask_rcnn_detection_model import get_maskrcnn_model


class ModelFactory:
    def __init__(self, model_name, image, image_size):
        self.model_name = model_name
        self.image = image
        self.image_size = image_size

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
            model = get_maskrcnn_model(self.image, self.image_size)
            return model
        else:
            raise ValueError("Model not implemented %s" % self.model_name)
