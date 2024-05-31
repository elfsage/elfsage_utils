def rename_layers(model, names, custom_objects=None):
    model_config = model.get_config()

    for layer in model_config['layers']:
        layer_name = layer['config']['name']

        if layer_name in names:
            layer['name'] = names[layer_name]
            layer['config']['name'] = names[layer_name]
            print('Renaming layer with name [{}] to [{}]'.format(layer_name, names[layer_name]))

        for node in (layer.get('inbound_nodes') or []):
            for input_layer in node:
                if input_layer[0] in names:
                    print('Renaming input of layer [{}] from [{}] to [{}]'.format(
                        layer['name'], input_layer[0], names[input_layer[0]]))
                    input_layer[0] = names[input_layer[0]]

    for layer in (model_config.get('input_layers') or []):
        if layer[0] in names:
            print('Renaming model input from [{}] to [{}]'.format(layer[0], names[layer[0]]))
            layer[0] = names[layer[0]]

    for layer in (model_config.get('output_layers') or []):
        if layer[0] in names:
            print('Renaming model output from [{}] to [{}]'.format(layer[0], names[layer[0]]))
            layer[0] = names[layer[0]]

    new_model = model.__class__.from_config(model_config, custom_objects=custom_objects)

    weights = [layer.get_weights() for layer in model.layers]
    for layer, weight in zip(new_model.layers, weights):
        layer.set_weights(weight)

    return new_model
