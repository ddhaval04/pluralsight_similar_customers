import pytest

from pluralsight_similar_customers.rest.app import app


@pytest.fixture
def client():
    context = app.app_context()
    context.push()
    yield app.test_client()
    context.pop()


def test_home(client):
    result = client.get("/")
    assert result.status_code == 200
