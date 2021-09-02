from dext.model.efficientdet.efficientdet_postprocess \
    import efficientdet_postprocess


class PostprocessorFactory:
    def __init__(self, model_name):
        self.model_name = model_name

    def factory(self):
        if "EFFICIENTDET" in self.model_name:
            return efficientdet_postprocess
        else:
            raise ValueError(
                "Preprocessor not implemented %s" % (self.model_name))
