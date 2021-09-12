import logging
import numpy as np

from paz.backend.image import resize_image
from paz.backend.image.opencv_image import write_image
from dext.inference.inference import inference_image
from dext.explainer.utils import get_saliency_mask
from dext.model.efficientdet.efficientdet_postprocess import process_outputs


LOGGER = logging.getLogger(__name__)


def check_saliency(model, model_name, raw_image, preprocessor_fn,
                   postprocessor_fn, image_size, saliency, box_index):
    modified_image = manipulate_raw_image_by_saliency(raw_image, saliency)
    forward_pass_outs = inference_image(model, modified_image, preprocessor_fn,
                                        postprocessor_fn, image_size)
    modified_detection_image = forward_pass_outs[0]
    write_image('modified_detections.jpg', modified_detection_image)
    outputs = forward_pass_outs[3]

    if "EFFICIENTDET" in model_name:
        outputs = process_outputs(outputs)
        for n, i in enumerate(box_index):
            LOGGER.info("Object confidence in same box of modified image: %s" %
                        outputs[0][int(i[0])][int(i[1] + 4)])


def manipulate_raw_image_by_saliency(raw_image, saliency,
                                     threshold=0.6):
    resized_raw_image = resize_image(
        raw_image, (saliency.shape))
    image = resized_raw_image.copy()
    mask_2d = get_saliency_mask(saliency.copy(), threshold)
    mask_3d = np.stack((mask_2d, mask_2d, mask_2d), axis=-1)
    result = np.where(mask_3d == 0, image, mask_3d).astype('uint8')
    return result


def manipulate_object_by_bg(raw_image, labels, object_info):
    pass
