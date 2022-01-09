from dext.interpretation_method.convert_box.ssd_conversion import (
    ssd_convert_coordinates)
from dext.interpretation_method.convert_box.efficientdet_conversion import (
    efficientdet_convert_coordinates)
from dext.interpretation_method.convert_box.fasterrcnn_conversion import (
    fasterrcnn_convert_coordinates)


def convert_to_image_coordinates(model_name, convouts, prior_boxes,
                                 visualize_index, image_size,
                                 image_scale, original_image_shape,
                                 to_ic=False):
    if model_name == 'SSD512':
        return ssd_convert_coordinates(convouts, prior_boxes, visualize_index,
                                       image_size, image_scale, to_ic)

    elif 'EFFICIENTDET' in model_name:
        return efficientdet_convert_coordinates(
            convouts, prior_boxes, visualize_index, image_size, image_scale,
            to_ic)

    elif model_name == 'FasterRCNN':
        return fasterrcnn_convert_coordinates(
            convouts, original_image_shape, visualize_index, image_size,
            image_scale, to_ic)

    elif 'MarineDebris' in model_name:
        return ssd_convert_coordinates(convouts, prior_boxes, visualize_index,
                                       image_size, image_scale, to_ic)
