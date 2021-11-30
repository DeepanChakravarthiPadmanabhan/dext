from dext.interpretation_method.convert_box.ssd_conversion import (
    ssd_convert_coordinates)
from dext.interpretation_method.convert_box.efficientdet_conversion import (
    efficientdet_convert_coordinates)
from dext.interpretation_method.convert_box.fasterrcnn_conversion import (
    fasterrcnn_convert_coordinates)


def convert_to_image_coordinates(model_name, convouts, prior_boxes,
                                 visualize_index, image_size,
                                 image_scale, original_image_shape):
    if model_name == 'SSD512':
        return ssd_convert_coordinates(convouts, prior_boxes, visualize_index,
                                       image_size, image_scale)

    elif 'EFFICIENTDET' in model_name:
        return efficientdet_convert_coordinates(
            convouts, prior_boxes, visualize_index, image_size, image_scale)

    elif model_name == 'FasterRCNN':
        return fasterrcnn_convert_coordinates(
            convouts, original_image_shape, visualize_index,
            image_size, image_scale)
