def inference_image(model, raw_image, preprocessor_fn,
                    postprocessor_fn, image_size):

    input_image, image_scales = preprocessor_fn(raw_image, image_size)
    # forward pass - get model outputs for input image
    class_outputs, box_outputs = model(input_image)
    detection_image, detections, box_index = postprocessor_fn(
        model, class_outputs, box_outputs, image_scales, raw_image)
    forward_pass_outs = (detection_image, detections,
                         box_index, class_outputs, box_outputs)
    return forward_pass_outs
