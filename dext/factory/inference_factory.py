from dext.inference.inference import inference_image_ssd
from dext.inference.inference import inference_image_efficientdet
from dext.inference.inference import inference_image_faster_rcnn


class InferenceFactory:
    def __init__(self, model_name):
        self.model_name = model_name

    def factory(self):
        if "EFFICIENTDET" in self.model_name:
            return inference_image_efficientdet
        elif "SSD" in self.model_name:
            return inference_image_ssd
        elif "FasterRCNN" in self.model_name:
            return inference_image_faster_rcnn
        elif "MarineDebris" == self.model_name:
            return inference_image_ssd
        else:
            raise ValueError(
                "Preprocessor not implemented %s" % self.model_name)
