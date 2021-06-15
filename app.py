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
CAT_BREED_CLASSIFIER = ''
DOG_BREED_CLASSIFIER = 'vgg_16_dog_breed_classifier'

CAT_BREEDS_PATH = 'info/cat_breeds.names'
DOG_BREED_PATH = 'info/dog_breeds.names'

SPECIES = ['Cat', 'Dog']
NBREEDS = (12, 25)
INPUT_SIZE = (224, 224, 3)


@st.cache(allow_output_mutation=True)
def load_species_classifier():
    model = load_model(SPECIES_CLASSIFIER)
    return model


@st.cache(allow_output_mutation=True)
def load_breed_classifier(species):
    if species.lower() == 'dog':
        return load_model(DOG_BREED_CLASSIFIER)
    elif species.lower() == 'cat':
        return None


@st.cache
def load_labels():
    with open(CAT_BREEDS_PATH) as f:
        cat_labels = f.readlines()
    with open(DOG_BREED_PATH) as f:
        dog_labels = f.readlines()

    return cat_labels, dog_labels


def predict_species(model, img):
    proba = model.predict(img).flatten()
    i = 1 if proba > .5 else 0
    return SPECIES[i]


def predict_breed(species, img):
    index = SPECIES.index(species)
    breed_proba = np.random.dirichlet(np.ones(NBREEDS[index]), size=1)[0]

    return breed_proba


def predict_breed_true(model, img):
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


def convert_to_predictable(img, resize=INPUT_SIZE[:2]):
    img = img.resize(size=resize)
    img = np.array(img)
    img = np.expand_dims(img, 0)

    return img


def main():
    # st.set_page_config(page_title='your_title', page_icon=favicon, layout='wide', initial_sidebar_state='auto')
    st.set_page_config(page_title='Pet Breed Classifier', layout='wide', page_icon=':dog:',
                       initial_sidebar_state='auto')

    # Sidebar
    st.sidebar.title('🐶 Pet Breed Classifier 🐱')
    st.sidebar.write("Take or upload a picture of your cat or dog to find out their predicted breed(s) !")
    uploaded_file = st.sidebar.file_uploader('', type=['jpeg', 'jpg', 'png'])

    # Loading species classifier model
    model = load_species_classifier()

    # load breed labels
    cat_labels, dog_labels = load_labels()

    # When a file is uploaded
    if uploaded_file is not None:
        # load image
        image = Image.open(uploaded_file)

        # Convert to numpy array with proper dimensions for prediction
        img_array = convert_to_predictable(image)

        # predict species
        species = predict_species(model, img_array)

        # loading breed predicting model
        breed_model = load_breed_classifier(species)

        # allow to change species
        species = st.sidebar.selectbox('Predicted species (click to change)', SPECIES, index=SPECIES.index(species))

        # predict breed
        breed_proba = predict_breed(species, image) if species.lower() == 'cat' else \
            predict_breed_true(breed_model, image)
        breed_index = breed_proba.argsort()[::-1]

        # obtain proper labels
        labels = cat_labels if species.lower() == 'cat' else dog_labels

        # # PRESENT RESULTS
        # SIDEBAR
        # st.sidebar.markdown("**Breed** : {}({:.0%})".format(labels[breed_index[0]],
        #                                                     breed_proba[breed_index[0]]))

        # MAIN

        # set max val to the number of breed probabilities it takes to each 99%
        max_val = (breed_proba[breed_index].cumsum() < .99).argmin() + 1

        col1, col2, col3 = st.beta_columns(3)

        col1.header(species)
        col1.subheader('Detected Breeds (Probability)')
        for i in range(max_val):
            label = labels[breed_index[i]]
            proba = breed_proba[breed_index[i]]
            col1.write("- {} {:.0%}".format(label,
                                            proba))
        col2.image(image, width=400)


if __name__ == '__main__':
    main()
