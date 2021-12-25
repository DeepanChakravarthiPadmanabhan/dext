from dext.model.efficientdet.utils import efficientdet_preprocess
from dext.model.ssd.utils import ssd_preprocess
from dext.model.faster_rcnn.faster_rcnn_preprocess import (
    faster_rcnn_preprocess)
from dext.model.marine_debris_ssd.utils import marine_debris_ssd_preprocess


class PreprocessorFactory:
    def __init__(self, model_name):
        self.model_name = model_name

    def factory(self):
        if "EFFICIENTDET" in self.model_name:
            return efficientdet_preprocess
        elif "SSD" in self.model_name:
            return ssd_preprocess
        elif "FasterRCNN" in self.model_name:
            return faster_rcnn_preprocess
        elif "MarineDebris" == self.model_name:
            return marine_debris_ssd_preprocess
        else:
            raise ValueError(
                "Preprocessor not implemented %s" % self.model_name)
