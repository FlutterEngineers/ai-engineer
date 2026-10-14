import uuid
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import app

client = TestClient(app)

# Example measurements from the educational dataset
VALID_FEATURES = [
    17.99,
    10.38,
    122.8,
    1001,
    0.1184,
    0.2776,
    0.3001,
    0.1471,
    0.2419,
    0.07871,
    1.095,
    0.9053,
    8.589,
    153.4,
    0.006399,
    0.04904,
    0.05373,
    0.01587,
    0.03003,
    0.006193,
    25.38,
    17.33,
    184.6,
    2019,
    0.1622,
    0.6656,
    0.7119,
    0.2654,
    0.4601,
    0.1189,
]


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "ML Prediction API is running"
    assert response.json()["expected_features"] == 30


def test_valid_prediction():
    response = client.post(
        "/predict",
        json={"features": VALID_FEATURES},
    )

    assert response.status_code == 200
    assert response.json()["predicted_class"] in ["malignant", "benign"]
    assert "probabilities" in response.json()

    data = response.json()

    # Check the prediction response
    assert "predicted_class" in data
    assert data["predicted_class"] in {"malignant", "benign"}

    # Check the model information
    assert "model" in data
    assert data["educational_use_only"] is True
    assert data["disclaimer"] == "Not a medical diagnostic tool."

    # Check prediction probabilities
    probabilities = data["probabilities"]
    assert set(probabilities.keys()) == {"malignant", "benign"}
    assert all(0 <= p <= 1 for p in probabilities.values())
    assert abs(sum(probabilities.values()) - 1.0) < 0.001


def test_rejects_wrong_number_of_features():
    response = client.post(
        "/predict",
        json={"features": [1, 2, 3]},
    )

    assert response.status_code == 422


def test_rejects_negative_features():
    features = VALID_FEATURES.copy()
    features[0] = -1

    response = client.post(
        "/predict",
        json={"features": features},
    )

    assert response.status_code == 422


def test_rejects_non_finite_features():
    # JSON does not support NaN or infinity as standard numeric values.
    # Send them as strings to check that the API rejects invalid types.
    for invalid_value in ("NaN", "Infinity", "-Infinity"):
        features = [1.0] * 30
        features[0] = invalid_value

        response = client.post(
            "/predict",
            json={"features": features},
        )

        assert response.status_code == 422


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["model"] == "Logistic Regression"


def test_request_id_header():
    response = client.get("/health")

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers

    request_id = response.headers["X-Request-ID"]

    # Verify that the request ID is a valid UUID
    uuid.UUID(request_id)


def test_request_ids_are_unique():
    first_response = client.get("/health")
    second_response = client.get("/health")

    first_id = first_response.headers["X-Request-ID"]
    second_id = second_response.headers["X-Request-ID"]

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_id != second_id


def test_prediction_handles_model_failure():
    with patch("api.model.predict", side_effect=RuntimeError("Internal model details")):
        response = client.post(
            "/predict",
            json={"features": VALID_FEATURES},
        )

    assert response.status_code == 500
    assert response.json() == {"detail": "Prediction failed. Please try again."}
    assert "Internal model details" not in response.text


def test_prediction_rate_limit():
    rate_limit_client = TestClient(app)

    responses = [
        rate_limit_client.post(
            "/predict",
            json={"features": VALID_FEATURES},
        )
        for _ in range(11)
    ]

    assert any(response.status_code == 429 for response in responses)
