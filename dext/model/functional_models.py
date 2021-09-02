import tensorflow as tf


def get_functional_efficientdet(model):
    image_size = model.image_size
    # Functional API calling only provides access to intermediate tensors
    original_dim = (image_size, image_size, 3)
    original_inputs = tf.keras.Input(shape=(original_dim), name="input")
    branch_tensors = model.backbone(original_inputs, False, True)
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
    functional_efficientdet = tf.keras.Model(
        inputs=original_inputs, outputs=[class_outputs, box_outputs])
    return functional_efficientdet


def get_functional_model(model_name, model):
    if "EFFICIENTDET" in model_name:
        return get_functional_efficientdet(model)
