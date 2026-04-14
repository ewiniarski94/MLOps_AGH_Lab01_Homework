from fastapi.testclient import TestClient
from app import app
from inference import load_sentiment_model
import pytest

client = TestClient(app)


@pytest.fixture(scope="module")
def model():
    return load_sentiment_model()


def test_model_inference_logic(model):
    samples = [
        ("The weather is very good!", "positive"),
        ("The weather is so bad.", "negative"),
        ("The weather is neutral.", "neutral"),
    ]
    for text, expected in samples:
        prediction = model.predict(text)
        assert prediction == expected
        assert isinstance(prediction, str)


def test_predict_success_valid_json():
    payload = {"text": "The weather is very good!"}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] == "positive"


def test_predict_empty_string():
    payload = {"text": ""}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
