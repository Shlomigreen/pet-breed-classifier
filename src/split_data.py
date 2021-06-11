import os
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import argparse


CATALOG_PATH = "info/images_catalog.csv"

def check_images(catalog):
    # Label catalog files as image / not image
    for i, row in catalog.iterrows():
        filename = row['full_path']
        try:
            im = Image.open(filename)
            catalog.loc[i, 'is_image'] = True
        except IOError:
            catalog.loc[i, 'is_image'] = False

    return catalog


def split_species(catalog, test_size=.1, random_state=42):
    # split indexes for species classifier
    X = catalog[catalog['is_image']].copy()
    X_train, X_test = train_test_split(X.index, test_size=test_size, random_state=random_state, stratify=X['species'])

    catalog.loc[X_train, 'species_train'] = True
    catalog.loc[X_test, 'species_train'] = False

    return catalog


def split_breed(catalog, species, test_size=.1, random_state=42):
    # split indexes for cat breed classifier
    X = catalog[catalog['is_image'] &
                ~catalog['breed'].isna() &
                (catalog['species'].str.lower() == species.lower())].copy()
    X_train, X_test = train_test_split(X.index, test_size=.1, random_state=42, stratify=X['breed'])

    catalog.loc[X_train, f'{species.lower()}_train'] = True
    catalog.loc[X_test, f'{species.lower()}_train'] = False


def move_file(filename, source_path, destination_path):
    os.makedirs(destination_path) if not os.path.exists(destination_path) else None
    os.replace(os.path.join(source_path, filename),
               os.path.join(destination_path, filename))


def clear_empty_dirs(root_dir='.'):
    for dirpath, dirs, files in os.walk(root_dir):
        if not dirs and not files:
            os.removedirs(dirpath)


def organize_species(df_info, undo=False):
    set_data = df_info.loc[~df_info['species_train'].isna(), :]

    for i, row in set_data.iterrows():
        row_set = 'train' if row['species_train'] else 'test'
        destination_dir = os.path.join('data', row_set, row['species'])

        if undo:
            move_file(row['file_name'],
                      destination_dir,
                      row['dir_path'])
        else:
            move_file(row['file_name'],
                      row['dir_path'],
                      destination_dir)

    clear_empty_dirs('data')


def organize_breed(df_info, species, undo=False):
    column = 'cat_train' if species.lower() == 'cat' else 'dog_train'

    set_data = df_info.loc[~df_info[column].isna(), :]

    for i, row in set_data.iterrows():
        row_set = 'train' if row[column] else 'test'
        destination_dir = os.path.join('data', row['species'].lower(), row_set, row['breed_name'])

        if undo:
            move_file(row['file_name'],
                      destination_dir,
                      row['dir_path'])

        else:
            move_file(row['file_name'],
                      row['dir_path'],
                      destination_dir)

    clear_empty_dirs('data')


def main():
    # catalog = pd.read_csv('info/images_catalog.csv')

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--validate", help="validate images in catalog; add new column with "
                                                 "boolean values to indicate if record at path is an image",
                        action="store_true")
    args = parser.parse_args()

    # FUNCTION_MAP = {'top20': my_top20_func,
    #                 'listapps': my_listapps_func}
    #
    # parser.add_argument('command', choices=FUNCTION_MAP.keys())
    #
    # args = parser.parse_args()
    #
    # func = FUNCTION_MAP[args.command]
    # func()

if __name__ == '__main__':
    main()
