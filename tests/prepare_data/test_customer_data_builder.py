import pytest

from pluralsight_similar_customers.prepare_data.customer_data_builder import (
    CustomerDataBuilder,
)


@pytest.fixture
def data_builder():
    data_builder = CustomerDataBuilder()

    return data_builder


def test_customer_data_builder_init(data_builder):
    assert data_builder.data_folder == "./data/"
    assert data_builder.data == {}
    assert data_builder.customer_summary_data == None
    assert data_builder.customer_list == []
