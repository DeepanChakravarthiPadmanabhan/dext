import tensorflow as tf
from dext.model.efficientdet.utils import create_multibox_head


def get_functional_efficientdet(model):
    image_size = model.image_size
    # Functional API calling only provides access to intermediate tensors
    input_shape = (image_size, image_size, 3)
    image = tf.keras.Input(shape=input_shape, name="image")
    branch_tensors = model.backbone(image, False, True)
    feature_levels = branch_tensors[model.min_level:model.max_level + 1]
    # Build additional input features that are not from backbone.
    for resample_layer in model.resample_layers:
        feature_levels.append(resample_layer(
            feature_levels[-1], False, None))
    # BiFPN layers
    fpn_features = model.fpn_cells(feature_levels, False)
    # Classification head
    class_outputs = model.class_net(fpn_features, False)
    # Box regression head
    box_outputs = model.box_net(fpn_features, False)
    outputs = create_multibox_head(
        class_outputs, box_outputs, model)
    functional_efficientdet = tf.keras.Model(
        inputs=image, outputs=outputs)
    return functional_efficientdet


def get_functional_model(model_name, model):
    if "EFFICIENTDET" in model_name:
        return get_functional_efficientdet(model)
