import os
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import sys

CATALOG_PATH = "info/catalog.csv"


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
    X_train, X_test = train_test_split(X.index, test_size=test_size, random_state=random_state, stratify=X['breed'])

    catalog.loc[X_train, f'{species.lower()}_train'] = True
    catalog.loc[X_test, f'{species.lower()}_train'] = False

    return catalog


def move_file(filename, source_path, destination_path):
    os.makedirs(destination_path) if not os.path.exists(destination_path) else None
    os.replace(os.path.join(source_path, filename),
               os.path.join(destination_path, filename))


def clear_empty_dirs(root_dir='.'):
    for dirpath, dirs, files in os.walk(root_dir):
        if not dirs and not files:
            os.removedirs(dirpath)


def organize_species(catalog, undo=False):
    set_data = catalog.loc[~catalog['species_train'].isna(), :]

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

    return catalog


def organize_breed(catalog, species, undo=False):
    column = 'cat_train' if species.lower() == 'cat' else 'dog_train'

    set_data = catalog.loc[~catalog[column].isna(), :]

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

    return catalog


def initialize_parser():
    parser = argparse.ArgumentParser(description='Split the data by breed or species with reorganization of folders',
                                     add_help=True)

    # Add acceptable arguments for CLI
    parser.add_argument('type', help='the type of split / organization functions to run',
                        choices=['breed', 'species'])
    parser.add_argument('-b', "--by", help="if chosen type is breed, a species [cat/dog] needs to be supplied as well",
                        required='breed' in sys.argv)
    parser.add_argument("-s", "--split", help="split and write set indicators to catalog",
                        action="store_true")
    parser.add_argument("-o", "--organize", help="also organize the folders based on splitting",
                        action="store_true")
    parser.add_argument("-u", "--undo", help="Undo organization of images; Restore original image location.",
                        action="store_true")

    args = parser.parse_args()

    return parser, args


def main():
    # Initiate parser and arguments
    parser, args = initialize_parser()

    # raise parser error if no operation [split or organize] was selected
    if not args.split and not args.organize:
        parser.error("no operation was selected; must indicate at least one of -s or -o")

    # read catalog file
    catalog = pd.read_csv(CATALOG_PATH)

    # run specific functions based on given arguments
    if args.type == 'breed':
        catalog = split_breed(catalog, args.by) if args.split else catalog
        catalog = organize_breed(catalog, args.by, undo=args.undo) if args.organize else catalog
    else:
        catalog = split_species(catalog) if args.split else catalog
        catalog = organize_species(catalog, undo=args.undo) if args.organize else catalog

    # update catalog file
    catalog.to_csv(CATALOG_PATH, index=False)


if __name__ == '__main__':
    main()
