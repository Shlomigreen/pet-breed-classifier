# 🐶 Pet Breed Classifier 🐱

**Used datasets:**
1. [Microsoft Cats vs Dogs Dataset](https://www.kaggle.com/shaunthesheep/microsoft-catsvsdogs-dataset)
2. [Cats and Dogs Breeds Classification Oxford Dataset](https://www.kaggle.com/zippyz/cats-and-dogs-breeds-classification-oxford-dataset)

**Model building process:**
1. Download required datasets 
2. Create a catalog file (pandas dataframe) to hold downloaded files information. 
3. Split & Re-organize files into train and test sets for each of the following:

    - Species based splitting : cat / dog classifier
    - Breed based splitting: cat breed classifier + dog breed classifier
    
4. Train and optimize each of the above described models using neural networks and transfer learning.
5. Classify any image using a web app integrated with the saved models.

---

# Clone the Repo 👨🏻‍🤝‍👨🏽
First, clone the github repo using the following command, and then navigate to the newly created directory:
```bash
git clone https://github.com/Shlomigreen/pet-breed-classifier
cd pet-breed-classifier
```


# Install Requirements ⚠️
In your chosen CLI, with python 3.8 or above installed, run the following command:

```bash
pip install -r requirements.txt
```

# Downloading Required Datasets 📥
Datasets for this project downloaded from Kaggle. An automated script to download all files
is existing in this project however you will first need to create a kaggle token:

1. Go to Kaggle
2. Login or create a new account
3. Navigate to Account > Create API Token and save the file somewhere safe on your computer.
4. Copy the file path (if working on Google Colab, upload the file first and then copy its path).

Run `src/download-files.py` with the path to `kaggle.json` token as follows:

```bash
python3 src/download-files.py <TOKEN PATH>
```

This will create (by default) a new directory named `data/` that holds both 
datasets in the original file tree system.

```
├── data
│   ├── cats-and-dogs-breeds-classification-oxford-dataset
│   │   ├── annotations
│   │   │   └── annotations
│   │   │       ├── trimaps
│   │   │       └── xmls
│   │   └── images
│   │       └── images
│   └── microsoft-catsvsdogs-dataset
│       └── PetImages
│           ├── Cat
│           └── Dog

```

# Creating a basic catalog 🗃️

```bash
python3 src/create-catalog.py
```

Included information per file:
- `dataset`: the source dataset (int).
- `species`: whether the file labeled as dog or cat in source datasets (str).
- `breed`: breed id as was given in source dataset (int).
- `breed_name`: literal name of the breed (string.title format).
- `dir_path`: relative path to the directory holding the file (str).
- `file_name`: name of the file, including extension (str).
- `full_path`: concatenation of `dir_path` and `file_name` (str).
- `is_image`: indicates of the found file can be opened as an image (boolean).


# Splitting data 📑 

Uses CLI in order to split files and reorganize them into train and test sets.
Split is done by adding new column(s) to catalog file.

```bash
python3 src/split_data.py
```


Added columns to catalog file:
- `species_train`
- `cat_train`
- `dog_train`
