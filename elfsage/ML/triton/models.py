def map_type(type_name):
    assert isinstance(type_name, str), 'Type name must be a string'

    type_map = {
        'bool': 'TYPE_BOOL',
        'uint8': 'TYPE_UINT8',
        'uint16': 'TYPE_UINT16',
        'uint32': 'TYPE_UINT32',
        'uint64': 'TYPE_UINT64',
        'int8': 'TYPE_INT8',
        'int16': 'TYPE_INT16',
        'int32': 'TYPE_INT32',
        'int64': 'TYPE_INT64',
        'float16': 'TYPE_FP16',
        'float32': 'TYPE_FP32',
        'float64': 'TYPE_FP64',
        'dtype(object)': 'TYPE_STRING',
    }

    return type_map[type_name.split('.')[-1]]


def keras_to_triton(model, model_name, parent_dir, model_version='1', batch_size=16):
    from proto import model_config_pb2
    import os

    model_dir = os.path.join(parent_dir, model_name)
    model_version_dir = os.path.join(model_dir, str(model_version), 'model.savedmodel')
    os.makedirs(model_version_dir, exist_ok=True)
    model.save(filepath=model_version_dir, overwrite=True, save_format='tf')

    model_inputs = []
    for i, keras_input in enumerate(model.inputs):
        model_input = model_config_pb2.ModelInput()
        model_input.name = keras_input._keras_history.layer.name
        model_input.data_type = map_type(keras_input.dtype.name)
        model_input.dims.extend(keras_input.shape[1:])
        model_inputs.append(model_input)

    model_outputs = []
    for i, keras_output in enumerate(model.outputs):
        model_output = model_config_pb2.ModelOutput()
        model_output.name = keras_output._keras_history.layer.name
        model_output.data_type = map_type(keras_output.dtype.name)
        model_output.dims.extend(keras_output.shape[1:])
        model_outputs.append(model_output)

    model_config = model_config_pb2.ModelConfig()
    model_config.name = model_name
    model_config.platform = 'tensorflow_savedmodel'
    model_config.max_batch_size = batch_size
    model_config.input.extend(model_inputs)
    model_config.output.extend(model_outputs)

    config_file_path = os.path.join(model_dir, 'config.pbtxt')
    with open(config_file_path, 'w', encoding='utf-8') as f:
        f.write(str(model_config))


def main():
    import tensorflow as tf
    from os import listdir
    from os.path import join, isdir
    from metrics import bce_dice_loss, dice_coeff, my_dice_coeff, my_dice_loss

    models_dir = r'E:\Python\PyCharmProjects\RESTAPI\models'
    triton_models_dir = 'E:\Python\PyCharmProjects\RESTAPI\models_triton'

    for model_name in listdir(models_dir):
        print(model_name)
        in_dir = join(models_dir, model_name)
        out_dir = join(triton_models_dir, model_name)

        if not isdir(in_dir):
            continue

        if isdir(out_dir):
            continue

        model = tf.keras.models.load_model(
            in_dir,
            {'bce_dice_loss': bce_dice_loss,
             'dice_coeff': dice_coeff,
             'my_dice_loss': my_dice_loss,
             'my_dice_coeff': my_dice_coeff}
        )

        print(model.outputs[-1].dtype)
        print(model.layers[-1].dtype)
        keras_to_triton(model, model_name, triton_models_dir)


if __name__ == '__main__':
    main()
