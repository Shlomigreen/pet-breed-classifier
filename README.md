# [🐶 Pet Breed Classifier 🐱](https://share.streamlit.io/shlomigreen/pet-breed-classifier/app.py)

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

All scripts in the project work by updating information into what's called a catalog file. 
The catalog is simply a `.csv` file that hold information about downloaded datasets files.
Most of the files supposed to be images, and they were collected to the catalog 
by a known path (as provided on each dataset's page).

<u>Included information (per file record):</u>

- `dataset`: the source dataset (int).
- `species`: whether the file labeled as dog or cat in source datasets (str).
- `breed`: breed id as was given in source dataset (int).
- `breed_name`: literal name of the breed (string.title format).
- `dir_path`: relative path to the directory holding the file (str).
- `file_name`: name of the file, including extension (str).
- `full_path`: concatenation of `dir_path` and `file_name` (str).

The catalog file will be created on the following path by default `info/catalog.csv` after running 
the catalog generating script:

```bash
python3 src/create-catalog.py
```

After the catalog has been created, a virtual pre-processing can be done to detect non-images and wrognly classified images. This will add a new column to the catalog file:
- `is_image`: indicates if the found file can be opened as an image and is truely labeled (boolean).

```bash
python3 src/pre-processing.py
```
> Note every file is being checked so this process take a bit of time.



# Splitting data 📑 

Uses CLI in order to split files and reorganize them into train and test sets.
Split is done by adding new column(s) to catalog file indicating for each model if the specific record 
is chosen to be used as train or test set.

```bash
python3 src/split_data.py [-h] [-b BY] [-s] [-o] [-u] {breed,species}
```

<u>There are 3 ways to split and organize the files:</u>
1. **For species classification (cat /dog)** : a boolean column named `species_train` will be created.
   <br> - split: `python3 src/split_data.py -s species`
   <br> - organize: `python3 src/split_data.py -o species`
   <br> - restore original file locations: `python3 src/split_data.py -o -u species`
   

2. **For cat breed classification** : a boolean column named `cat_train` will be created.
   <br> - split: `python3 src/split_data.py -s -b cat breed `
   <br> - organize: `python3 src/split_data.py -o -b cat breed`
   <br> - restore original file locations: `python3 src/split_data.py -o -u -b cat breed`
   

3. **For dog breed classification** : a boolean column named `dog_train` will be created.
   <br> - split: `python3 src/split_data.py -s -b cat breed `
   <br> - organize: `python3 src/split_data.py -o -b cat breed`
   <br> - restore original file locations: `python3 src/split_data.py -o -u -b cat breed`


When running a split and organize commands on a either species breed, new sub-folders will be created
under the `data` directory including the species name, train, test and a directory for each breed.
Example of directory tree after running a split and organize commands on cat breeds:

```angular2html
data/
├── cat
│   ├── test
│   │   ├── Abyssinian
│   │   ├── Bengal
│   │   ├── Birman
│   │   ├── Bombay
│   │   ├── British Shorthair
│   │   ├── Egyptian Mau
│   │   ├── Maine Coon
│   │   ├── Persian
│   │   ├── Ragdoll
│   │   ├── Russian Blue
│   │   ├── Siamese
│   │   └── Sphynx
│   └── train
│       ├── Abyssinian
│       ├── Bengal
│       ├── Birman
│       ├── Bombay
│       ├── British Shorthair
│       ├── Egyptian Mau
│       ├── Maine Coon
│       ├── Persian
│       ├── Ragdoll
│       ├── Russian Blue
│       ├── Siamese
│       └── Sphynx
```

> Note that when cloning the repo, split is already done in all three ways and its information
> is contained in the catalog file using the default `test_size=.1 random_state=42'`.


> > If needed, used functions from each command can be imported to a notebook / other python script:
> <br>```from src.split_data import split_species, split_breed, organize_species, organize_breed```
