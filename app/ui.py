# streamlit run app/ui.py
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

SPECIES = ['Cat', 'Dog']
NBREEDS = (25, 12)


@st.cache
def predict_species(img):
    species = np.random.choice(SPECIES)
    return species


@st.cache
def predict_breed(species, img):
    index = SPECIES.index(species)
    breed_proba = np.random.dirichlet(np.ones(NBREEDS[index]), size=1)[0]

    return breed_proba


def radar_chart(top_n, breed_proba):
    top_n_breed_index = breed_proba.argsort()[::-1][:top_n]

    df = pd.DataFrame(dict(
        probability=breed_proba[top_n_breed_index].round(3),
        breed=top_n_breed_index.astype('str')
    )
    )
    fig = px.line_polar(df, r='probability',
                        theta='breed',
                        line_close=True,
                        width=400, height=400)
    # fig.update_layout(
    #     margin=dict(l=0, r=0, t=0, b=0),
    # )
    st.write(fig)


# st.set_page_config(page_title='your_title', page_icon=favicon, layout='wide', initial_sidebar_state='auto')
st.set_page_config(page_title='Pet Breed Classifier', layout='wide', initial_sidebar_state='auto')


# Sidebar
st.sidebar.title('🐶 Pet Breed Classifier 🐱')
st.sidebar.write("Take or upload a picture of your cat or dog to find out their predicted breed(s) !")
uploaded_file = st.sidebar.file_uploader('', type=['jpeg', 'jpg', 'png'])


# When a file is uploaded
if uploaded_file is not None:
    # load image
    image = Image.open(uploaded_file)

    # predict species
    species = predict_species(image)

    # allow to change species
    species = st.sidebar.selectbox('Predicted species (click to change)', SPECIES, index=SPECIES.index(species))

    # predict breed
    breed_proba = predict_breed(species, image)
    breed_index = breed_proba.argmax()

    # write results
    st.sidebar.markdown("**Breed** - {}".format(breed_index))

    # show uploaded image
    st.image(image, caption='', width=200)

    # show radar plot
    val = st.sidebar.slider('Top Breed Similarity', min_value=3, max_value=10)
    radar_chart(top_n=val, breed_proba=breed_proba)
