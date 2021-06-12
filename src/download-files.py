import os
import sys
import logging
from zipfile import ZipFile

KAGGLE_TOKEN_PATH = sys.argv[1]
DATA_PATH = os.path.join(os.getcwd(), 'data')

# Setup logger
logger = logging.getLogger('logger')
logger.setLevel(logging.DEBUG)

# Create Formatter
formatter = logging.Formatter('%(asctime)s \t | %(levelname)s \t | %(message)s')

# # create a file handler and add it to logger
# file_handler = logging.FileHandler('car_log_file.log')
# file_handler.setLevel(logging.DEBUG)
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

# Steam handler to print logs to the screen
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def copy_kaggle_token(token_path=KAGGLE_TOKEN_PATH):
    if not (os.path.exists(token_path)):
        raise FileExistsError(f"Could not find kaggle token `kaggle.json` in given path: {token_path}")
    else:
        logger.info("Found `kaggle.json` file.")

    # Transfer kaggle API token to its needed location and change permissions
    kaggle_dir = os.path.expanduser('~/.kaggle')
    if not os.path.exists(kaggle_dir):
        os.mkdir(kaggle_dir)
        logger.debug("Created ~/.kaggle directory")
    os.system('cp {} ~/.kaggle/'.format(token_path))
    logger.debug("Copied `kaggle.json` file")
    os.system('chmod 600 ~/.kaggle/kaggle.json')


def unzip(zip_path, out_dir=""):
    """Unzips a file at given path to desired folder."""
    with ZipFile(zip_path, 'r') as zipObj:
        zipObj.extractall(out_dir)


def download_dataset(dataset_tuple):
    """Downloads dataset from kaggle from given tuple of format: (username, setname)"""
    dataset = "/".join(dataset_tuple)
    os.system(f'kaggle datasets download {dataset} -p {DATA_PATH}')



def download_datasets():
    logger.info("Downloading datasets")
    datasets_list = [('shaunthesheep', 'microsoft-catsvsdogs-dataset'),
                     ('zippyz', 'cats-and-dogs-breeds-classification-oxford-dataset')]

    for dataset_tuple in datasets_list:
        zip_path = os.path.join(DATA_PATH, dataset_tuple[1])

        if not os.path.exists(zip_path):
            logger.debug(f"Downloading {dataset_tuple}")
            download_dataset(dataset_tuple)

            logger.debug("Unzipping...")
            unzip(zip_path + ".zip", zip_path)

            logger.debug("Removing zip files...")
            os.chmod(zip_path, 0o777)
            os.remove(zip_path+".zip")
        else:
            logger.debug(f"'{zip_path}' exists ; skipping")

    logger.info("Downloading datasets completed")


copy_kaggle_token()
download_datasets()
