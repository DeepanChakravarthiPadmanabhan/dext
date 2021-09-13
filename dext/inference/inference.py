def inference_image_efficientdet(model, raw_image, preprocessor_fn,
                                 postprocessor_fn, image_size):

    input_image, image_scales = preprocessor_fn(raw_image, image_size)
    # forward pass - get model outputs for input image
    outputs = model(input_image)
    detection_image, detections, box_index = postprocessor_fn(
        model, outputs, image_scales, raw_image)
    forward_pass_outs = (detection_image, detections,
                         box_index, outputs)
    return forward_pass_outs


def inference_image_ssd(model, raw_image, preprocessor_fn,
                        postprocessor_fn, image_size):

    input_image = preprocessor_fn(raw_image, model.input_shape[1:3])
    # forward pass - get model outputs for input image
    outputs = model(input_image)
    detection_image, detections, box_index = postprocessor_fn(
        model, outputs, raw_image)
    forward_pass_outs = (detection_image, detections,
                         box_index, outputs)
    return forward_pass_outs


class InferenceFactory:
    def __init__(self, model_name):
        self.model_name = model_name

    def factory(self):
        if "EFFICIENTDET" in self.model_name:
            return inference_image_efficientdet
        elif "SSD" in self.model_name:
            return inference_image_ssd
        else:
            raise ValueError(
                "Preprocessor not implemented %s" % self.model_name)
