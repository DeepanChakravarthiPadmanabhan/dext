from dext.model.efficientdet.utils import efficientdet_preprocess


class PreprocessorFactory:
    def __init__(self, model_name):
        self.model_name = model_name

    def factory(self):
        if "EFFICIENTDET" in self.model_name:
            return efficientdet_preprocess
        else:
            raise ValueError(
                "Preprocessor not implemented %s" % (self.model_name))
