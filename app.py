# streamlit run app.py
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from src.modeling import load_model
import urllib.request
import os
from image_classification import predict_species, predict_breed

# Constants
SPECIES_CLASSIFIER = 'vgg16_species_classifier'
BREED_CLASSIFIERS = {'cat': 'vgg_16_cat_breed_classifier',
                     'dog': 'vgg_16_dog_breed_classifier'}

GITHUB_REPO = 'https://github.com/Shlomigreen/pet-breed-classifier/'
RELEASE_TAG = 'v1.0'

CAT_BREEDS_PATH = 'info/cat_breeds.names'
DOG_BREED_PATH = 'info/dog_breeds.names'

SPECIES = ['Cat', 'Dog']
NBREEDS = (12, 25)


def download_model(tag, name, *extensions):
    base_url = GITHUB_REPO + r'releases/download/{}/{}.{}'

    for extn in extensions:
        url = base_url.format(tag, name, extn)
        filename = url.split('/')[-1]

        save_path = 'models/{}'.format(filename)
        if not os.path.exists(save_path):
            urllib.request.urlretrieve(url, save_path)


def download_models(tag):
    if not os.path.exists('models'):
        os.mkdir('models')
    download_model(tag, SPECIES_CLASSIFIER, 'weights', 'conf')
    for _, value in BREED_CLASSIFIERS.items():
        download_model(tag, value, 'weights', 'conf')


def page_setup():
    # st.set_page_config(page_title='your_title', page_icon=favicon, layout='wide', initial_sidebar_state='auto')
    st.set_page_config(page_title='Pet Breed Classifier', layout='wide', page_icon=':dog:',
                       initial_sidebar_state='auto')

    # download_required models
    download_models(RELEASE_TAG)

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


def convert_to_predictable(img, resize):
    img = img.resize(size=resize)
    img = np.array(img)
    img = np.expand_dims(img, 0)

    return img


# def predict_species(model, img):
#     img = convert_to_predictable(img, model.input_shape[1:3])
#
#     proba = model.predict(img).flatten()
#     i = 0 if proba < 0.5 else 1
#
#     species = SPECIES[i]
#
#     print("Predicted species:", species)
#
#     return species


# @st.cache(allow_output_mutation=True)
def load_breed_classifier(species):
    model_name = BREED_CLASSIFIERS[species]
    model = load_model(model_name)
    return model


# def predict_breed(species, img):
#     index = SPECIES.index(species)
#     breed_proba = np.random.dirichlet(np.ones(NBREEDS[index]), size=1)[0]
#
#     return breed_proba


# def predict_breed(model, img):
#     img_array = convert_to_predictable(img, model.input_shape[1:3])
#
#     breed_proba = model.predict(img_array)
#     return breed_proba.flatten()


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
    st.write("Use the sidebar to upload an image")

    # # Loading species classifier model
    # species_model = load_species_classifier()

    # load breed labels
    breed_labels = load_labels()

    # When a file is uploaded
    if uploaded_file is not None:
        # load uploaded file as image
        image = Image.open(uploaded_file)

        # # predict species
        # predicted_species = predict_species(species_model, image)
        predicted_species = predict_species(image)

        # allow to change species in case of mis-classification
        species = st.sidebar.selectbox('Predicted species (click to change)',
                                       SPECIES,
                                       index=SPECIES.index(predicted_species))

        #if species != predicted_species:
            #print("Predicted species changed to", species)

        # converting species to lowercase for future use
        species = species.lower()

        # # loading specific breed predicting model
        # breed_model = load_breed_classifier(species)

        # predict breed
        breed_proba = predict_breed(species, image)
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

        INFO_URL = 'https://dogtime.com/dog-breeds/{}'

        for i in range(max_val):
            label = labels[breed_index[i]]
            proba = breed_proba[breed_index[i]]
            text = "- {} ({:.0%})".format(label,
                                          proba)
            if species == 'dog':
                breed = label.lower().replace(' ', '-')
                breed_link = INFO_URL.format(breed)
                col1.markdown("{} [ℹ️]({})".format(text, breed_link))
                #print(text)
            else:
                col1.write(text)
                #print(text)
        col2.image(image, width=400)


if __name__ == '__main__':
    main()
