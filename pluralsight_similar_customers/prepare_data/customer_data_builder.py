import os

import numpy as np
import pandas as pd

from pluralsight_similar_customers.shared.logger import get_logger

logger = get_logger(__name__)


class CustomerDataBuilder(object):
    def __init__(self, data_folder="./data/"):
        """
        Args:
            data_folder (str): Path specifying the input data folder.
        """
        if not os.path.isdir(data_folder):
            raise ValueError("{dir} doesn't exist.".format(dir=data_folder))
        self.data_folder = data_folder
        self.data = {}
        self.customer_summary_data = None
        self.customer_list = []

    def load_data_from_dir(self):
        """Function for reading all the csv files present in the `self.data_folder` directory
        """
        try:
            filenames = os.listdir(self.data_folder)
            for filename in filenames:
                if filename.endswith(".csv"):
                    path = os.path.join(self.data_folder, filename)
                    key = filename.strip().split(".")[0]
                    self.data[key] = pd.read_csv(path)
        except Exception as exception:
            logger.error(exception)
            raise ValueError("Error reading the data.")

    def build_dataset(self):
        """Builds the training dataset for model training.
        """
        self.processed_dataset = self._get_customer_course_view_interaction()

    def _get_customer_course_view_interaction(self):
        """Builds a customer-course_view interaction (time spent) matrix for
        performing matrix-factorization in the next step.

        Returns:
            user_course_interaction_matrix_demeaned [np.ndarray]: A numpy matrix
                of customer-course_view interaction
        """
        try:
            user_course_view_tags = pd.merge(
                self.data["user_course_views"],
                self.data["course_tags"],
                how="inner",
                on="course_id",
            )
            user_course_views_avg = user_course_view_tags.groupby(
                ["user_handle", "course_tags"], as_index=False
            )["view_time_seconds"].mean()
            user_course_interaction = user_course_views_avg.pivot(
                index="user_handle", columns="course_tags", values="view_time_seconds"
            ).fillna(0)
            user_course_interaction_matrix = user_course_interaction.values
            user_course_view_mean = np.mean(user_course_interaction_matrix, axis=1)
            user_course_interaction_matrix_demeaned = (
                user_course_interaction_matrix - user_course_view_mean.reshape(-1, 1)
            )
            self.customer_list = user_course_interaction.index.tolist()
            self.customer_summary_data = user_course_views_avg

            return user_course_interaction_matrix_demeaned
        except Exception as exception:
            logger.error(exception)
            raise ValueError(
                "Error building the customer-course-view interaction matrix!"
            )
