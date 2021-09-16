def get_layers_and_length(layer):
    if hasattr(layer, 'layers'):
        block_length = len(layer.layers)
        block_layers = layer.layers
    else:
        block_length = 0
        block_layers = layer
    return block_length, block_layers


def get_all_layers(model):
    all_layers = []
    for i in model.layers[1:]:
        block_length, block_layers = get_layers_and_length(i)
        if block_length:
            all_layers.extend(block_layers)
        else:
            all_layers.append(block_layers)
    return all_layers
