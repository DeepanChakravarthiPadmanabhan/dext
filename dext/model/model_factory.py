from dext.model.efficientdet.efficientdet import EFFICIENTDETD0, EFFICIENTDETD1
from dext.model.efficientdet.efficientdet import EFFICIENTDETD2, EFFICIENTDETD3
from dext.model.efficientdet.efficientdet import EFFICIENTDETD4, EFFICIENTDETD5
from dext.model.efficientdet.efficientdet import EFFICIENTDETD6, EFFICIENTDETD7
from dext.model.efficientdet.efficientdet import EFFICIENTDETD7x

class ModelFactory:
    def __init__(self, model_name):
        self.model_name = model_name

    def factory(self):
        if self.model_name == "EFFICIENTDETD0":
            return EFFICIENTDETD0
        elif self.model_name == "EFFICIENTDETD1":
            return EFFICIENTDETD1
        elif self.model_name == "EFFICIENTDETD2":
            return EFFICIENTDETD2
        elif self.model_name == "EFFICIENTDETD3":
            return EFFICIENTDETD3
        elif self.model_name == "EFFICIENTDETD4":
            return EFFICIENTDETD4
        elif self.model_name == "EFFICIENTDETD5":
            return EFFICIENTDETD5
        elif self.model_name == "EFFICIENTDETD6":
            return EFFICIENTDETD6
        elif self.model_name == "EFFICIENTDETD7":
            return EFFICIENTDETD7
        elif self.model_name == "EFFICIENTDETD7x":
            return EFFICIENTDETD7x
        else:
            raise ValueError("Model not implemented %s" % (self.model_name))