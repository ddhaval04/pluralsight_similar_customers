import numpy as np
import pytest

from pluralsight_similar_customers.models.customer_similarity_model import (
    CustomerSimilarityModel,
)


@pytest.fixture
def untrained_customer_similarity_model():
    model = CustomerSimilarityModel()

    return model


@pytest.fixture
def trained_customer_similarity_model():
    model = CustomerSimilarityModel()
    model.load_models_from_files()

    return model


def test_untrained_customer_similarity_model_init(untrained_customer_similarity_model):
    assert untrained_customer_similarity_model.is_trained == False


def test_customer_similarity_model_predict_without_train(
    untrained_customer_similarity_model,
):
    input_embedding = np.ones((1, 50))
    with pytest.raises(RuntimeError):
        untrained_customer_similarity_model.predict(input_embedding)
