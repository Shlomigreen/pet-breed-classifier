import cv2
from PIL import Image
import pandas as pd

CATALOG_PATH = 'info/catalog.csv'


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


def mark_bad(catalog):
    bad = [('5673.jpg', 'Cat'),
           ('4688.jpg', 'Cat'),
           ('12376.jpg', 'Dog'),
           ('1043.jpg', 'Dog'),
           ('8456.jpg', 'Cat'),
           ('9517.jpg', 'Dog'),
           ('7377.jpg', 'Cat'),
           ('1773.jpg', 'Dog'),
           ('8736.jpg', 'Dog'),
           ('10712.jpg', 'Cat'),
           ('7564.jpg', 'Cat')]

    bad_index = []

    for img_name, species_name in bad:
        img_path = catalog[(catalog['species'] == species_name) & (catalog['file_name'] == img_name)]['full_path']
        idx = img_path.index

        bad_index.append(idx[0])

    catalog.loc[bad_index, 'is_image'] = False

    return catalog


def main():
    catalog = pd.read_csv(CATALOG_PATH)

    # Add is image label
    catalog = check_images(catalog)

    # mark bad images
    catalog = mark_bad(catalog)

    # rewrite catalog
    catalog.to_csv(CATALOG_PATH)


if __name__ == '__main__':
    main()
