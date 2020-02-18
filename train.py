from pluralsight_similar_customers.prepare_data.customer_data_builder import (
    CustomerDataBuilder,
)
from pluralsight_similar_customers.models.customer_similarity_model import (
    CustomerSimilarityModel,
)


def train():
    obj = CustomerDataBuilder()
    obj.load_data_from_dir()
    obj.build_dataset()
    obj = CustomerSimilarityModel()
    obj.train()


if __name__ == "__main__":
    train()
