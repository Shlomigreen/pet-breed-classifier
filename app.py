# streamlit run app.py
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from src.modeling import load_model
import matplotlib.pyplot as plt

# Constants
SPECIES_CLASSIFIER = 'vgg16_species_classifier'
CAT_BREED_CLASSIFIER = 'vgg_16_cat_breed_classifier'
DOG_BREED_CLASSIFIER = 'vgg_16_dog_breed_classifier'

BREED_CLASSIFIERS = {'cat': 'vgg_16_cat_breed_classifier',
                     'dog': 'vgg_16_dog_breed_classifier'}

CAT_BREEDS_PATH = 'info/cat_breeds.names'
DOG_BREED_PATH = 'info/dog_breeds.names'

SPECIES = ['Cat', 'Dog']
NBREEDS = (12, 25)
INPUT_SIZE = (224, 224, 3)


def page_setup():
    # st.set_page_config(page_title='your_title', page_icon=favicon, layout='wide', initial_sidebar_state='auto')
    st.set_page_config(page_title='Pet Breed Classifier', layout='wide', page_icon=':dog:',
                       initial_sidebar_state='auto')

    # Sidebar
    st.sidebar.title('🐶 Pet Breed Classifier 🐱')
    st.sidebar.write("Take or upload a picture of your cat or dog to find out their predicted breed(s) !")

    # Add an upload file sidebar
    uploaded_file = st.sidebar.file_uploader('', type=['jpeg', 'jpg', 'png'])

    return uploaded_file


@st.cache(allow_output_mutation=True)
def load_species_classifier():
    model = load_model(SPECIES_CLASSIFIER)
    return model


@st.cache
def load_labels():
    with open(CAT_BREEDS_PATH) as f:
        cat_labels = f.readlines()
    with open(DOG_BREED_PATH) as f:
        dog_labels = f.readlines()

    labels = {'cat': cat_labels,
              'dog': dog_labels}

    return labels


def convert_to_predictable(img, resize=INPUT_SIZE[:2]):
    img = img.resize(size=resize)
    img = np.array(img)
    img = np.expand_dims(img, 0)

    return img


def predict_species(model, img):
    img = convert_to_predictable(img, model.input_shape[1:3])

    proba = model.predict(img).flatten()
    i = 0 if proba < 0.5 else 1

    return SPECIES[i]


@st.cache(allow_output_mutation=True)
def load_breed_classifier(species):
    model_name = BREED_CLASSIFIERS[species]
    return load_model(model_name)


# def predict_breed(species, img):
#     index = SPECIES.index(species)
#     breed_proba = np.random.dirichlet(np.ones(NBREEDS[index]), size=1)[0]
#
#     return breed_proba


def predict_breed(model, img):
    img_array = convert_to_predictable(img, model.input_shape[1:3])

    breed_proba = model.predict(img_array)
    return breed_proba.flatten()


def radar_chart(top_n, breed_proba, labels):
    top_n_breed_index = breed_proba.argsort()[::-1][:top_n]
    df = pd.DataFrame(dict(
        probability=breed_proba[top_n_breed_index].round(3),
        breed=np.array(labels)[top_n_breed_index]
    )
    )
    fig = px.line_polar(df, r='probability',
                        theta='breed',
                        line_close=True,
                        width=400, height=400)
    # fig.update_layout(
    #     margin=dict(l=0, r=0, t=0, b=0),
    # )
    return fig


def main():
    uploaded_file = page_setup()

    # Loading species classifier model
    species_model = load_species_classifier()

    # load breed labels
    breed_labels = load_labels()

    # When a file is uploaded
    if uploaded_file is not None:
        # load uploaded file as image
        image = Image.open(uploaded_file)

        # predict species
        species = predict_species(species_model, image)

        # allow to change species in case of mis-classification
        species = st.sidebar.selectbox('Predicted species (click to change)', SPECIES, index=SPECIES.index(species))

        # converting species to lowercase for future use
        species = species.lower()

        # loading specific breed predicting model
        breed_model = load_breed_classifier(species)

        # predict breed
        breed_proba = predict_breed(breed_model, image)
        breed_index = breed_proba.argsort()[::-1]

        # obtain proper labels
        labels = breed_labels[species]

        # # PRESENT RESULTS
        # SIDEBAR
        # st.sidebar.markdown("**Breed** : {}({:.0%})".format(labels[breed_index[0]],
        #                                                     breed_proba[breed_index[0]]))

        # MAIN

        # set max val to the number of breed probabilities it takes to each 99%
        max_val = (breed_proba[breed_index].cumsum() < .99).argmin() + 1

        col1, col2, col3 = st.beta_columns(3)

        col1.header(species.title())
        col1.subheader('Detected Breeds')
        col1.write("Shown breeds cover 99% of prediction probabilities:")

        for i in range(max_val):
            label = labels[breed_index[i]]
            proba = breed_proba[breed_index[i]]
            col1.write("- {} ({:.0%})".format(label,
                                            proba))
        col2.image(image, width=400)


if __name__ == '__main__':
    main()
