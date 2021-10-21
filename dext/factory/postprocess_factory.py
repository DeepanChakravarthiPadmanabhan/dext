from dext.model.efficientdet.efficientdet_postprocess import (
    efficientdet_postprocess)
from dext.model.ssd.ssd_postprocess import ssd_postprocess
from dext.model.faster_rcnn.faster_rcnn_postprocess import (
    faster_rcnn_postprocess)


class PostprocessorFactory:
    def __init__(self, model_name):
        self.model_name = model_name

    def factory(self):
        if "EFFICIENTDET" in self.model_name:
            return efficientdet_postprocess
        elif "SSD" in self.model_name:
            return ssd_postprocess
        elif "FasterRCNN" in self.model_name:
            return faster_rcnn_postprocess
        else:
            raise ValueError(
                "Preprocessor not implemented %s" % self.model_name)
