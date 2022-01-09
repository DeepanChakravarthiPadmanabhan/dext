from dext.model.faster_rcnn.utils import norm_boxes_graph
from dext.utils.get_image import get_image


def inference_image_efficientdet(model, raw_image_path, preprocessor_fn,
                                 postprocessor_fn, image_size,
                                 explain_top5_backgrounds=False,
                                 load_type='rgb'):
    raw_image = get_image(raw_image_path, load_type)
    input_image, image_scales = preprocessor_fn(raw_image, image_size)
    outputs = model(input_image)
    detection_image, detections, box_index = postprocessor_fn(
        model, outputs, image_scales, raw_image, image_size,
        explain_top5_backgrounds)
    forward_pass_outs = (detection_image, detections, box_index, outputs)
    return forward_pass_outs


def inference_image_ssd(model, raw_image_path, preprocessor_fn,
                        postprocessor_fn, image_size,
                        explain_top5_backgrounds=False, load_type='rgb'):
    raw_image = get_image(raw_image_path, load_type)
    input_image, image_scales = preprocessor_fn(raw_image, image_size)
    outputs = model(input_image)
    detection_image, detections, box_index = postprocessor_fn(
        model, outputs, image_scales, raw_image, image_size,
        explain_top5_backgrounds)
    forward_pass_outs = (detection_image, detections, box_index, outputs)
    return forward_pass_outs


def inference_image_faster_rcnn(model, raw_image_path, preprocessor_fn,
                                postprocessor_fn, image_size,
                                explain_top5_backgrounds=False,
                                load_type='rgb'):
    raw_image = get_image(raw_image_path, load_type)
    input_image, image_scales = preprocessor_fn(raw_image, image_size)
    config_window = norm_boxes_graph(image_scales, (image_size, image_size))
    outputs = model(input_image, config_window)
    detection_image, detections, box_index = postprocessor_fn(
        model, outputs, image_scales, raw_image, image_size,
        explain_top5_backgrounds)
    forward_pass_outs = (detection_image, detections, box_index,
                         outputs)
    return forward_pass_outs


def inference_image_marine_debris_ssd_resnet20(
        model, raw_image_path, preprocessor_fn, postprocessor_fn, image_size,
        explain_top5_backgrounds=False, load_type='gray'):
    raw_image = get_image(raw_image_path, load_type)
    input_image, image_scales = preprocessor_fn(raw_image, image_size)
    outputs = model(input_image)
    detection_image, detections, box_index = postprocessor_fn(
        model, outputs, image_scales, raw_image, image_size,
        explain_top5_backgrounds)
    forward_pass_outs = (detection_image, detections, box_index, outputs)
    return forward_pass_outs


def inference_image_marine_debris_ssd_mobilenet(
        model, raw_image_path, preprocessor_fn, postprocessor_fn, image_size,
        explain_top5_backgrounds=False, load_type='gray'):
    raw_image = get_image(raw_image_path, load_type)
    input_image, image_scales = preprocessor_fn(raw_image, image_size)
    outputs = model(input_image)
    detection_image, detections, box_index = postprocessor_fn(
        model, outputs, image_scales, raw_image, image_size,
        explain_top5_backgrounds)
    forward_pass_outs = (detection_image, detections, box_index, outputs)
    return forward_pass_outs
