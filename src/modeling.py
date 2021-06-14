import pickle
import json
import tensorflow.keras.models
import os


def validate_dirs(*dir_paths):
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


def save_model(model, name, conf_path='models/config', weight_path='models/weights'):
    validate_dirs(conf_path, weight_path)

    # Saving the architecture
    with open(os.path.join(conf_path, name + '.conf'), 'w') as outfile:
        config = model.get_config()
        json.dump(config, outfile)

    # Saving the weights
    with open(os.path.join(weight_path, name + '.weights'), "wb") as output_file:
        weights = model.get_weights()
        pickle.dump(weights, output_file)


def load_model(name, conf_path='models/config', weight_path='models/weights'):
    # load architecture
    with open(os.path.join(conf_path, name + '.conf')) as json_file:
        loaded_config = json.load(json_file)

    # load weights
    with open(os.path.join(weight_path, name + '.weights'), 'rb') as weights_file:
        loaded_weights = pickle.load(weights_file)

    model = tensorflow.keras.Sequential().from_config(loaded_config)
    model.set_weights(loaded_weights)

    return model


def main():
    pass


if __name__ == '__main__':
    main()
