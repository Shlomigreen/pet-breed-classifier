from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import pickle
import json
import tensorflow.keras.models
import numpy as np
import os


class Model:
    def __init__(self, name, input_shape, batch_size=64):
        self.name = name
        self.input_shape = input_shape  # tuple, 3 values
        self.batch_size = batch_size

        self.image_generator = None
        self.dir_iterators = {}
        self.callbacks = []
        pass

    def create_generator(self, **kwargs):
        img_gen = ImageDataGenerator(**kwargs)
        self.image_generator = img_gen

    def flow_from_directory(self, name, directory,
                            target_size=None,
                            batch_size=None,
                            class_mode=None, subset=None, **kwargs):
        if target_size is None:
            target_size = self.input_shape[:2]
        if batch_size is None:
            batch_size = self.batch_size

        dir_iter = self.image_generator.flow_from_directory(directory=directory, target_size=target_size,
                                                            batch_size=batch_size,
                                                            class_mode=class_mode,
                                                            subset=subset, **kwargs)
        self.dir_iterators[name] = dir_iter

    def configurate(self, sequence):
        self.model = Sequential(sequence)

    def compile(self, optimizer='adam', loss='binary_crossentropy', metrics='accuracy', **kwargs):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

    def add_callback(self, callback):
        self.callbacks.add(callback)

    def fit(self, **kwargs):
        self.train_log = self.model.fit(callbacks=self.callbacks, **kwargs)

    def save_model(self, conf_path='models/config', weight_path='models/weights'):
        # Saving the architecture
        with open(os.path.join(conf_path, self.name + '.conf'), 'w') as outfile:
            config = self.model.get_config()
            json.dump(config, outfile)

        # Saving the weights
        with open(os.path.join(weight_path, self.name + '.weights'), "wb") as output_file:
            weights = self.model.get_weights()
            pickle.dump(weights, output_file)

    def load_model(self, conf_path='models/config', weight_path='models/weights'):
        # load architecture
        with open(os.path.join(conf_path, self.name + '.conf')) as json_file:
            loaded_config = json.load(json_file)

        # load weights
        with open(os.path.join(weight_path, self.name + '.weights'), 'rb') as weights_file:
            loaded_weights = pickle.load(weights_file)

        self.model = tensorflow.keras.Sequential().from_config(loaded_config)
        self.model.set_weights(loaded_weights)
