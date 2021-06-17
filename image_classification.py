from src.modeling import load_model
import numpy as np
import tensorflow as tf

SPECIES_CLASSIFIER = 'vgg16_species_classifier'
BREED_CLASSIFIERS = {'cat': 'vgg_16_cat_breed_classifier',
                     'dog': 'vgg_16_dog_breed_classifier'}

SPECIES = ['Cat', 'Dog']


def load_species_classifier():
    model = load_model(SPECIES_CLASSIFIER)
    return model


def convert_to_predictable(img, resize):
    img = img.resize(size=resize)
    img = np.array(img)
    img = np.expand_dims(img, 0)

    return img


#@tf.function(experimental_relax_shapes=True)
def predict_species(img):
    model = load_species_classifier()

    img = convert_to_predictable(img, model.input_shape[1:3])

    proba = model.predict(img).flatten()
    i = 0 if proba < 0.5 else 1

    species = SPECIES[i]

    print("Predicted species:", species)

    return species


def load_breed_classifier(species):
    model_name = BREED_CLASSIFIERS[species]
    model = load_model(model_name)
    return model


#@tf.function(experimental_relax_shapes=True)
def predict_breed(species, img):
    model = load_breed_classifier(species)
    img_array = convert_to_predictable(img, model.input_shape[1:3])

    breed_proba = model.predict(img_array)
    return breed_proba.flatten()
