import os
import pickle

from pluralsight_similar_customers.shared.logger import get_logger

logging = get_logger(__name__)


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_to_pickle(path, data):
    try:
        with open(path, "wb") as file_handle:
            pickle.dump(data, file_handle)
        logging.info("Successfully saved data to {path}".format(path=path))
    except pickle.PickleError as error:
        logging.error(error)


def read_from_pickle(path):
    try:
        logging.info("Reading data from {path}".format(path=path))
        with open(path, "rb") as file_handle:
            data = pickle.load(file_handle)

        return data
    except pickle.PickleError as error:
        logging.error(error)
