import pandas as pd
import os
from PIL import Image

# PARENT_DIR = os.path.dirname(os.getcwd())
PARENT_DIR = ''
DATA_PATH = 'data'
OUTPUT_DIR = 'info'
OUTPUT_NAME = 'catalog.csv'

DF = pd.DataFrame(columns=['dataset',
                           'species',
                           'breed',
                           'breed_name',
                           'dir_path',
                           'file_name',
                           'full_path'])


def catalog_microsoft_dataset():
    microsoft_dataset = DF.copy()

    dir_name = 'microsoft-catsvsdogs-dataset/PetImages'
    dir_path = os.path.join(DATA_PATH, dir_name)
    sub_dirs = ['Cat', 'Dog']

    for sub_dir in sub_dirs:
        sub_path = os.path.join(dir_path, sub_dir)
        entry = pd.DataFrame({'dataset': 1,
                              'species': sub_dir,
                              'breed': None,
                              'dir_path': sub_path,
                              'file_name': os.listdir(os.path.join(PARENT_DIR, sub_path))})

        microsoft_dataset = microsoft_dataset.append(entry, ignore_index=True)

    print("Microsoft dataset shape:", microsoft_dataset.shape)

    return microsoft_dataset


def catalog_oxford_dataset():
    dir_name = 'cats-and-dogs-breeds-classification-oxford-dataset'
    dir_path = os.path.join(DATA_PATH, dir_name)

    # loading list txt file into a dataframe
    oxford_dataset = pd.read_csv(os.path.join(PARENT_DIR, dir_path, 'annotations', 'annotations', 'list.txt'),
                                 skiprows=[i for i in range(0, 6)], sep=" ",
                                 header=None)
    oxford_dataset.columns = ['image', 'class_id', 'species', 'breed_id']

    # mapping species to actual labels
    oxford_dataset['species'] = oxford_dataset['species'].map({1: 'Cat', 2: 'Dog'})

    # creating a dataframe to be merged with other datasets
    # file names were added with '.jpg' extension
    image_path = os.path.join(dir_path, 'images', 'images')
    oxfort_dataset_formatted = pd.DataFrame({'dataset': 2,
                                             'species': oxford_dataset['species'],
                                             'breed': oxford_dataset['class_id'],
                                             'dir_path': image_path,
                                             'file_name': oxford_dataset['image'] + '.jpg'})

    print("Oxford dataset shape:", oxfort_dataset_formatted.shape)

    return oxfort_dataset_formatted


def main():
    # Merging datasets
    microsoft_dataset = catalog_microsoft_dataset()
    oxfort_dataset_formatted = catalog_oxford_dataset()

    df = DF.copy()
    df = df.append(microsoft_dataset)
    df = df.append(oxfort_dataset_formatted)

    # Convert file name to breed name
    df['breed_name'] = df['file_name'].str.extract(r'(.*)_\d*.jpg', expand=False)
    df['breed_name'] = df['breed_name'].str.replace('_', ' ')
    df['breed_name'] = df['breed_name'].str.title()

    # Adding absolute path
    df['full_path'] = (df['dir_path'] + '/' + df['file_name'])

    # Resetting index to have a running number on rows
    df.reset_index(drop=True, inplace=True)

    # Saving catalog to file
    if not os.path.exists(os.path.join(PARENT_DIR, OUTPUT_DIR)):
        os.mkdir(os.path.join(PARENT_DIR, OUTPUT_DIR))

    df.to_csv(os.path.join(PARENT_DIR, OUTPUT_DIR, OUTPUT_NAME), index=False)


if __name__ == '__main__':
    main()
