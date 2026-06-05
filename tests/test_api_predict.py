"""Tests for Flask API endpoints."""
import pytest
from src.api.app import create_app, decode_image_from_base64
import numpy as np
import base64
import cv2


@pytest.fixture
def client():
    """Create Flask test client."""
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_index(client):
    """Test index route returns HTML."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'<!DOCTYPE html>' in response.data or b'<html' in response.data


def test_predict_missing_data(client):
    """Test /predict with missing image data."""
    response = client.post('/predict', json={})
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data


def test_predict_invalid_base64(client):
    """Test /predict with invalid base64."""
    response = client.post('/predict', json={'data': 'invalid base64!!!'})
    assert response.status_code == 400


def test_predict_valid_request(client):
    """Test /predict with valid image data."""
    # Create a minimal JPEG image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, encoded = cv2.imencode('.jpg', img)
    b64 = base64.b64encode(encoded).decode('utf-8')

    response = client.post('/predict', json={'data': b64})
    
    # Should return 200 and valid metrics
    assert response.status_code == 200
    data = response.get_json()
    
    # Verify response structure
    required_keys = ['ear_left', 'ear_right', 'mar', 'pitch', 'yaw', 'roll', 'perclos', 'confidence']
    for key in required_keys:
        assert key in data, f"Missing key: {key}"
        assert isinstance(data[key], (int, float)), f"{key} should be numeric"
        assert 0.0 <= data[key] <= 1.0 or key in ['pitch', 'yaw', 'roll'], f"{key} should be in valid range"


def test_predict_data_url_format(client):
    """Test /predict with data: URL format."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, encoded = cv2.imencode('.jpg', img)
    b64 = base64.b64encode(encoded).decode('utf-8')
    data_url = f'data:image/jpeg;base64,{b64}'

    response = client.post('/predict', json={'data': data_url})
    
    assert response.status_code == 200
    data = response.get_json()
    assert 'confidence' in data


def test_decode_image_from_base64():
    """Test image decoding utility."""
    # Create test image
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    _, encoded = cv2.imencode('.jpg', img)
    b64 = base64.b64encode(encoded).decode('utf-8')
    
    # Test decoding
    decoded = decode_image_from_base64(b64)
    assert decoded is not None
    assert decoded.shape[0] == 100
    assert decoded.shape[1] == 100
    assert decoded.shape[2] == 3


def test_decode_image_with_data_url():
    """Test decoding with data: URL prefix."""
    img = np.ones((50, 50, 3), dtype=np.uint8) * 64
    _, encoded = cv2.imencode('.jpg', img)
    b64 = base64.b64encode(encoded).decode('utf-8')
    data_url = f'data:image/jpeg;base64,{b64}'
    
    decoded = decode_image_from_base64(data_url)
    assert decoded is not None
    assert decoded.shape == (50, 50, 3)


def test_decode_image_invalid_input():
    """Test decoding with invalid input."""
    result = decode_image_from_base64('invalid!!!')
    assert result is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
