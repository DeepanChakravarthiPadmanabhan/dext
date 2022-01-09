import gin
from paz import processors as pr
from paz.abstract import SequentialProcessor, Processor
from dext.model.utils import ResizeImage, find_image_scale
from paz.models.detection.ssd300 import SSD300
from dext.model.ssd.ssd_postprocess import nms_per_class, filterboxes
from dext.model.ssd.ssd_postprocess import denormalize_boxes
from dext.model.ssd.ssd_postprocess import draw_bounding_boxes
from dext.model.ssd.ssd_postprocess import get_top5_bg_ssd
from dext.utils.class_names import get_classes


def marine_debris_ssd_vgg16_preprocess(image, image_size, only_resize=False):
    input_image_shape = image.shape
    if type(image_size) == int:
        image_size = (image_size, image_size)
    if only_resize:
        preprocessing = SequentialProcessor([
            ResizeImage(image_size),
            pr.CastImage(float),
            pr.ExpandDims(axis=0)])
    else:
        preprocessing = SequentialProcessor([
            ResizeImage(image_size),
            pr.CastImage(float),
            pr.NormalizeImage(),
            pr.ExpandDims(axis=0)])
    image = preprocessing(image)
    processed_image_shape = image.shape
    image_scale = find_image_scale(input_image_shape, processed_image_shape)
    return image, image_scale


@gin.configurable
def marine_debris_ssd_vgg16(weight_path):
    model = SSD300(12, None, None)
    model.load_weights(weight_path)
    return model


def marine_debris_ssd_vgg16_postprocess(
        model, outputs, image_scale, raw_image, image_size=512,
        explain_top5_backgrounds=False):
    class_names = get_classes('MarineDebris', 'MarineDebris')
    postprocess = SequentialProcessor([pr.Squeeze(axis=None),
                                       pr.DecodeBoxes(model.prior_boxes)])
    detections = postprocess(outputs)
    detections = nms_per_class(detections, nms_thresh=0.4, conf_thresh=0.01)
    detections, class_map_idx = filterboxes(
        detections, class_names, conf_thresh=0.4)
    detections = denormalize_boxes(detections, model.input_shape[1:3],
                                   image_scale)
    image = draw_bounding_boxes(
        raw_image, detections, class_names, max_size=image_size)

    if explain_top5_backgrounds:
        image, detections, class_map_idx = get_top5_bg_ssd(
            model, outputs, image_scale, raw_image)

    return image, detections, class_map_idx
