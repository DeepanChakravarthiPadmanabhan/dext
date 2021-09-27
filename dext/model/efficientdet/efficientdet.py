from dext.model.efficientdet.efficientdet_model import EfficientDet
from dext.utils.constants import EFFICIENTDET_WEIGHT_PATH


WEIGHT_PATH = EFFICIENTDET_WEIGHT_PATH


def EFFICIENTDET(image_size, num_classes, fpn_num_filters,
                 fpn_cell_repeats, box_class_repeats, anchor_scale,
                 min_level, max_level, fpn_weight_method, return_base,
                 model_name, BACKBONE):
    """ Generates an EfficientDet model with the parameter
    values passed as argument.
    # Arguments
        model_name: A string of EfficientDet model name.
        backbone: A string of EfficientNet backbone name used
        in EfficientDet.
        image_size: Int, size of the input image
        fpn_num_filters: Int, FPN filter output size
        fpn_cell_repeats: Int, Number of consecutive FPN block
        box_class_repeats: Int, Number of consective regression
        and classification blocks
        anchor_scale: Int, specifying the number of anchor
        scales
        min_level: Int, minimum level for features.
        max_level: Int, maximum level for features.
        fpn_weight_method: A string specifying the feature
        fusion weighting method in fpn
    # Returns
        model: EfficientDet model
    """

    model = EfficientDet(image_size, num_classes, fpn_num_filters,
                         fpn_cell_repeats, box_class_repeats, anchor_scale,
                         min_level, max_level, fpn_weight_method, return_base,
                         model_name, BACKBONE)
    weights_path = WEIGHT_PATH + model_name + '.h5'
    model.build((1, image_size, image_size, 3))
    model.summary()
    model.load_weights(weights_path)
    return model


def EFFICIENTDETD0(image_size=512, num_classes=90, fpn_num_filters=64,
                   fpn_cell_repeats=3, box_class_repeats=3, anchor_scale=4.0,
                   min_level=3, max_level=7, fpn_weight_method='fastattention',
                   return_base=False, model_name='efficientdet-d0',
                   BACKBONE='efficientnet-b0'):
    """ Instantiates EfficientDet-D0 model with the default
    setting provided in the official implementation.
    # Arguments
        model_name: A string of EfficientDet model name.
        backbone: A string of EfficientNet backbone name used
        in EfficientDet.
        image_size: Int, size of the input image
        fpn_num_filters: Int, FPN filter output size
        fpn_cell_repeats: Int, Number of consecutive FPN block
        box_class_repeats: Int, Number of consective regression
        and classification blocks
        anchor_scale: Int, specifying the number of anchor
        scales
        min_level: Int, minimum level for features.
        max_level: Int, maximum level for features.
        fpn_weight_method: A string specifying the feature
        fusion weighting method in fpn
    # Returns
        model: EfficientDet-D0 model
    """
    model = EFFICIENTDET(image_size, num_classes, fpn_num_filters,
                         fpn_cell_repeats, box_class_repeats, anchor_scale,
                         min_level, max_level, fpn_weight_method, return_base,
                         model_name, BACKBONE)
    return model


def EFFICIENTDETD1(image_size=640, num_classes=90,  fpn_num_filters=88,
                   fpn_cell_repeats=4, box_class_repeats=3, anchor_scale=4.0,
                   min_level=3, max_level=7, fpn_weight_method='fastattention',
                   return_base=False, model_name='efficientdet-d1',
                   BACKBONE='efficientnet-b1'):
    """ Instantiates EfficientDet-D1 model with the default
    setting provided in the official implementation.
    # Arguments
        model_name: A string of EfficientDet model name.
        backbone: A string of EfficientNet backbone name used
        in EfficientDet.
        image_size: Int, size of the input image
        fpn_num_filters: Int, FPN filter output size
        fpn_cell_repeats: Int, Number of consecutive FPN block
        box_class_repeats: Int, Number of consective regression
        and classification blocks
        anchor_scale: Int, specifying the number of anchor
        scales
        min_level: Int, minimum level for features.
        max_level: Int, maximum level for features.
        fpn_weight_method: A string specifying the feature
        fusion weighting method in fpn
    # Returns
        model: EfficientDet-D1 model
    """
    model = EFFICIENTDET(image_size, num_classes, fpn_num_filters,
                         fpn_cell_repeats, box_class_repeats, anchor_scale,
                         min_level, max_level, fpn_weight_method, return_base,
                         model_name, BACKBONE)
    return model


def EFFICIENTDETD2(image_size=768, num_classes=90, fpn_num_filters=112,
                   fpn_cell_repeats=5, box_class_repeats=3, anchor_scale=4.0,
                   min_level=3, max_level=7, fpn_weight_method='fastattention',
                   return_base=False, model_name='efficientdet-d2',
                   BACKBONE='efficientnet-b2'):
    """ Instantiate EfficientDet-D2 model with the default
    setting provided in the official implementation.
    # Arguments
        model_name: A string of EfficientDet model name.
        backbone: A string of EfficientNet backbone name used
        in EfficientDet.
        image_size: Int, size of the input image
        fpn_num_filters: Int, FPN filter output size
        fpn_cell_repeats: Int, Number of consecutive FPN block
        box_class_repeats: Int, Number of consective regression
        and classification blocks
        anchor_scale: Int, specifying the number of anchor
        scales
        min_level: Int, minimum level for features.
        max_level: Int, maximum level for features.
        fpn_weight_method: A string specifying the feature
        fusion weighting method in fpn
    # Returns
        model: EfficientDet-D2 model
    """
    model = EFFICIENTDET(image_size, num_classes, fpn_num_filters,
                         fpn_cell_repeats, box_class_repeats, anchor_scale,
                         min_level, max_level, fpn_weight_method, return_base,
                         model_name, BACKBONE)
    return model


def EFFICIENTDETD3(image_size=896, num_classes=90, fpn_num_filters=160,
                   fpn_cell_repeats=6, box_class_repeats=4, anchor_scale=4.0,
                   min_level=3, max_level=7, fpn_weight_method='fastattention',
                   return_base=False, model_name='efficientdet-d3',
                   BACKBONE='efficientnet-b3'):
    """ Instantiates EfficientDet-D3 model with the default
    setting provided in the official implementation.
    # Arguments
        model_name: A string of EfficientDet model name.
        backbone: A string of EfficientNet backbone name used
        in EfficientDet.
        image_size: Int, size of the input image
        fpn_num_filters: Int, FPN filter output size
        fpn_cell_repeats: Int, Number of consecutive FPN block
        box_class_repeats: Int, Number of consective regression
        and classification blocks
        anchor_scale: Int, specifying the number of anchor
        scales
        min_level: Int, minimum level for features.
        max_level: Int, maximum level for features.
        fpn_weight_method: A string specifying the feature
        fusion weighting method in fpn
    # Returns
        model: EfficientDet-D3 model
    """
    model = EFFICIENTDET(image_size, num_classes, fpn_num_filters,
                         fpn_cell_repeats, box_class_repeats, anchor_scale,
                         min_level, max_level, fpn_weight_method, return_base,
                         model_name, BACKBONE)
    return model


def EFFICIENTDETD4(image_size=1024, num_classes=90, fpn_num_filters=224,
                   fpn_cell_repeats=7, box_class_repeats=4, anchor_scale=4.0,
                   min_level=3, max_level=7, fpn_weight_method='fastattention',
                   return_base=False, model_name='efficientdet-d4',
                   BACKBONE='efficientnet-b4'):
    """ Instantiates EfficientDet-D4 model with the default
    setting provided in the official implementation.
    # Arguments
        model_name: A string of EfficientDet model name.
        backbone: A string of EfficientNet backbone name used
        in EfficientDet.
        image_size: Int, size of the input image
        fpn_num_filters: Int, FPN filter output size
        fpn_cell_repeats: Int, Number of consecutive FPN block
        box_class_repeats: Int, Number of consective regression
        and classification blocks
        anchor_scale: Int, specifying the number of anchor
        scales
        min_level: Int, minimum level for features.
        max_level: Int, maximum level for features.
        fpn_weight_method: A string specifying the feature
        fusion weighting method in fpn
    # Returns
        model: EfficientDet-D4 model
    """
    model = EFFICIENTDET(image_size, num_classes, fpn_num_filters,
                         fpn_cell_repeats, box_class_repeats, anchor_scale,
                         min_level, max_level, fpn_weight_method, return_base,
                         model_name, BACKBONE)
    return model


def EFFICIENTDETD5(image_size=1280, num_classes=90, fpn_num_filters=288,
                   fpn_cell_repeats=7, box_class_repeats=4, anchor_scale=4.0,
                   min_level=3, max_level=7, fpn_weight_method='fastattention',
                   return_base=False, model_name='efficientdet-d5',
                   BACKBONE='efficientnet-b5'):
    """ Instantiates EfficientDet-D5 model with the default
    setting provided in the official implementation.
    # Arguments
        model_name: A string of EfficientDet model name.
        backbone: A string of EfficientNet backbone name used
        in EfficientDet.
        image_size: Int, size of the input image
        fpn_num_filters: Int, FPN filter output size
        fpn_cell_repeats: Int, Number of consecutive FPN block
        box_class_repeats: Int, Number of consective regression
        and classification blocks
        anchor_scale: Int, specifying the number of anchor
        scales
        min_level: Int, minimum level for features.
        max_level: Int, maximum level for features.
        fpn_weight_method: A string specifying the feature
        fusion weighting method in fpn
    # Returns
        model: EfficientDet-D5 model
    """
    model = EFFICIENTDET(image_size, num_classes, fpn_num_filters,
                         fpn_cell_repeats, box_class_repeats, anchor_scale,
                         min_level, max_level, fpn_weight_method, return_base,
                         model_name, BACKBONE)
    return model


def EFFICIENTDETD6(image_size=1280, num_classes=90, fpn_num_filters=384,
                   fpn_cell_repeats=8, box_class_repeats=5, anchor_scale=5.0,
                   min_level=3, max_level=7, fpn_weight_method='sum',
                   return_base=False, model_name='efficientdet-d6',
                   BACKBONE='efficientnet-b6'):
    """ Instantiates EfficientDet-D6 model with the default
    setting provided in the official implementation.
    # Arguments
        model_name: A string of EfficientDet model name.
        backbone: A string of EfficientNet backbone name used
        in EfficientDet.
        image_size: Int, size of the input image
        fpn_num_filters: Int, FPN filter output size
        fpn_cell_repeats: Int, Number of consecutive FPN block
        box_class_repeats: Int, Number of consective regression
        and classification blocks
        anchor_scale: Int, specifying the number of anchor
        scales
        min_level: Int, minimum level for features.
        max_level: Int, maximum level for features.
        fpn_weight_method: A string specifying the feature
        fusion weighting method in fpn
    # Returns
        model: EfficientDet-D6 model
    """
    model = EFFICIENTDET(image_size, num_classes, fpn_num_filters,
                         fpn_cell_repeats, box_class_repeats, anchor_scale,
                         min_level, max_level, fpn_weight_method, return_base,
                         model_name, BACKBONE)
    return model


def EFFICIENTDETD7(image_size=1536, num_classes=90, fpn_num_filters=384,
                   fpn_cell_repeats=8, box_class_repeats=5, anchor_scale=5.0,
                   min_level=3, max_level=7, fpn_weight_method='sum',
                   return_base=False, model_name='efficientdet-d7',
                   BACKBONE='efficientnet-b6'):
    """ Instantiates EfficientDet-D7 model with the default
    setting provided in the official implementation.
    # Arguments
        model_name: A string of EfficientDet model name.
        backbone: A string of EfficientNet backbone name used
        in EfficientDet.
        image_size: Int, size of the input image
        fpn_num_filters: Int, FPN filter output size
        fpn_cell_repeats: Int, Number of consecutive FPN block
        box_class_repeats: Int, Number of consective regression
        and classification blocks
        anchor_scale: Int, specifying the number of anchor scales
        min_level: Int, minimum level for features.
        max_level: Int, maximum level for features.
        fpn_weight_method: A string specifying the feature
        fusion weighting method in fpn
    # Returns
        model: EfficientDet-D7 model
    """
    model = EFFICIENTDET(image_size, num_classes, fpn_num_filters,
                         fpn_cell_repeats, box_class_repeats, anchor_scale,
                         min_level, max_level, fpn_weight_method, return_base,
                         model_name, BACKBONE)
    return model


def EFFICIENTDETD7x(image_size=1536, num_classes=90, fpn_num_filters=384,
                    fpn_cell_repeats=8, box_class_repeats=5, anchor_scale=4.0,
                    min_level=3, max_level=8, fpn_weight_method='sum',
                    return_base=False, model_name='efficientdet-d7x',
                    BACKBONE='efficientnet-b7'):
    """ Instantiates EfficientDet-D7x model with the default
    setting provided in the official implementation.
    # Arguments
        model_name: A string of EfficientDet model name.
        backbone: A string of EfficientNet backbone name used
        in EfficientDet.
        image_size: Int, size of the input image
        fpn_num_filters: Int, FPN filter output size
        fpn_cell_repeats: Int, Number of consecutive FPN block
        box_class_repeats: Int, Number of consective regression
        and classification blocks
        anchor_scale: Int, specifying the number of anchor
        scales
        min_level: Int, minimum level for features.
        max_level: Int, maximum level for features.
        fpn_weight_method: A string specifying the feature
        fusion weighting method in fpn
    # Returns
        model: EfficientDet-D7x model
    """
    model = EFFICIENTDET(image_size, num_classes, fpn_num_filters,
                         fpn_cell_repeats, box_class_repeats, anchor_scale,
                         min_level, max_level, fpn_weight_method, return_base,
                         model_name, BACKBONE)
    return model
