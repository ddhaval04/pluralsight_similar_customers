import os

from scipy.sparse.linalg import svds
from sklearn.neighbors import NearestNeighbors

from pluralsight_similar_customers.prepare_data.customer_data_builder import (
    CustomerDataBuilder,
)
from pluralsight_similar_customers.shared.model import Model
from pluralsight_similar_customers.shared.logger import get_logger
from pluralsight_similar_customers.shared.util import (
    create_dir,
    read_from_pickle,
    write_to_pickle,
)

logging = get_logger(__name__)


class CustomerSimilarityModel(Model):
    is_trained = False
    models_folder = "./trained_models/"
    models_filename = os.path.join(models_folder, "customer_similarity_knn.pkl")
    _index_metadata_filename = os.path.join(
        models_folder, "customer_similarity_metadata.pkl"
    )

    def __init__(self):
        create_dir(self.models_folder)
        if not self.is_trained:
            self.customer_data_builder = CustomerDataBuilder()
            self.customer_data_builder.load_data_from_dir()
            self.customer_data_builder.build_dataset()

    def load_models_from_files(self):
        """Loads models and metadata present in self.models_folder. (Make sure the files are pickle.)
        """
        self.index = read_from_pickle(self.models_filename)
        self.index_metadata = read_from_pickle(self._index_metadata_filename)
        self.is_trained = True

    def train(self):
        """Performs matrix factorization on the prcessed_dataset obtained in the previous step. Then uses the customer embeddings, and builds knn and pickles it to the `self.models_folder`.
        """
        logging.info("Performing SVD ...")
        U, _, _ = svds(self.customer_data_builder.processed_dataset, k=50)
        # sigma = np.diag(sigma)
        self.customer_embeddings = U
        self._build_index()
        self.is_trained = True

    def _build_index(self):
        """Builds nearest neighbors and saves them to `self.models_folder`.
        """
        logging.info("Building indexes ...")
        self.index = NearestNeighbors(n_neighbors=10, metric="cosine")
        self.index.fit(self.customer_embeddings)
        logging.info("Saving indexes to disk ...")
        write_to_pickle(self.models_filename, self.index)
        self.index_metadata = {}
        self.index_metadata["customer_list"] = self.customer_data_builder.customer_list
        self.index_metadata["customer_embeddings"] = self.customer_embeddings
        self.index_metadata[
            "customer_summary"
        ] = self.customer_data_builder.customer_summary_data
        logging.info("Saving metadata to disk ...")
        write_to_pickle(self._index_metadata_filename, self.index_metadata)

    def predict(self, data):
        """Performs a nearest neighbor search for the query embedding.
        Args:
            data (np.array): A numpy array of the customer embedding.

        Returns:
            summary (pd.DataFrame): Summary of similar customers.
        """
        if self.is_trained:
            neighbors_index = self.index.kneighbors(
                data, n_neighbors=15, return_distance=False
            )
            nearest_customers = [
                self.index_metadata["customer_list"][idx] for idx in neighbors_index[0]
            ]
            summary = self.index_metadata["customer_summary"].loc[
                self.index_metadata["customer_summary"].user_handle.isin(
                    nearest_customers
                )
            ]

            return summary
        else:
            raise RuntimeError(
                "Model is not trained. First train the model and then predict!"
            )

    def predict_on_batch(self, data_batch):
        pass
