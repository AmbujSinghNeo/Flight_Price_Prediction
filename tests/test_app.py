import sys
import os
from fastapi.testclient import TestClient
import pytest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from main import app, get_model_pipeline

# --- Test Fixtures ---
@pytest.fixture
def valid_payload():
    """Standard valid payload for testing."""
    return {
        "airline": "Vistara", "source_city": "Delhi", "departure_time": "Morning",
        "stops": "one", "arrival_time": "Night", "destination_city": "Mumbai",
        "class": "Business", "duration": 15.83, "days_left": 26
    }

# --- Test Cases ---
client = TestClient(app)

def test_read_root():
    """Test the root endpoint to ensure it's running."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_successful_prediction(valid_payload):
    """Test a successful prediction using the actual loaded model."""
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert data["predicted_price"] > 0

def test_model_unavailable(valid_payload):
    """Test API behavior by overriding the model dependency to return None."""
    def override_get_model_unavailable():
        return None
    
    app.dependency_overrides[get_model_pipeline] = override_get_model_unavailable
    
    # Recreate client to use the override
    with TestClient(app) as client:
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 503
        assert "model is currently unavailable" in response.json()["detail"].lower()

    app.dependency_overrides.clear() # Clean up the override

def test_model_prediction_error_handling(valid_payload):
    """Test API's 500 error handling by mocking a prediction failure."""
    mock_pipeline = MagicMock()
    mock_pipeline.predict.side_effect = Exception("A low-level prediction error")
    
    def override_get_model_with_error():
        return mock_pipeline
        
    app.dependency_overrides[get_model_pipeline] = override_get_model_with_error
    
    with TestClient(app) as client:
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 500
        assert "internal error" in response.json()["detail"].lower()
        
    app.dependency_overrides.clear()

def test_model_returns_invalid_prediction(valid_payload):
    """Test API's 400 error handling for invalid model outputs."""
    mock_pipeline = MagicMock()
    mock_pipeline.predict.return_value = [-10.0]
    
    def override_get_model_with_invalid_output():
        return mock_pipeline

    app.dependency_overrides[get_model_pipeline] = override_get_model_with_invalid_output

    with TestClient(app) as client:
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 400
        assert "invalid (non-positive or non-finite) price" in response.json()["detail"].lower()
    
    app.dependency_overrides.clear()

def test_validation_error_same_city(valid_payload):
    """Test Pydantic validation for source and destination cities."""
    payload = valid_payload.copy()
    payload["destination_city"] = "Delhi"
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "source and destination cities must be different" in response.json()["detail"].lower()
