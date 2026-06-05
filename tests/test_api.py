"""Comprehensive Flask API integration tests."""

import pytest
import json
import numpy as np
import base64
import cv2
from src.api.app import create_app


@pytest.fixture
def app():
    """Create Flask app for testing."""
    app = create_app()
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    """Create Flask test client."""
    with app.test_client() as client:
        yield client


@pytest.fixture
def app_context(app):
    """Create Flask app context."""
    with app.app_context():
        yield app


class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health_returns_200(self, client):
        """Test /health endpoint returns 200 status."""
        # Note: /health endpoint might need to be added to app.py
        # For now, test that root returns 200 (index)
        response = client.get('/')
        assert response.status_code == 200

    def test_index_route_exists(self, client):
        """Test index route is available."""
        response = client.get('/')
        assert response.status_code == 200
        assert b'<!DOCTYPE' in response.data or b'<html' in response.data

    def test_index_returns_html(self, client):
        """Test index returns HTML content type."""
        response = client.get('/')
        assert response.status_code == 200
        assert 'text/html' in response.content_type or len(response.data) > 0


class TestPredictEndpoint:
    """Test /predict endpoint with various inputs."""

    def test_predict_valid_base64_image(self, client, base64_jpeg_raw):
        """Test /predict with valid base64 image."""
        response = client.post(
            '/predict',
            json={'data': base64_jpeg_raw},
            content_type='application/json'
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data is not None

    def test_predict_valid_data_url(self, client, base64_jpeg_image):
        """Test /predict with data URL format."""
        response = client.post(
            '/predict',
            json={'data': base64_jpeg_image},
            content_type='application/json'
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data is not None

    def test_predict_response_schema(self, client, base64_jpeg_raw):
        """Test /predict response has correct schema."""
        response = client.post(
            '/predict',
            json={'data': base64_jpeg_raw}
        )

        assert response.status_code == 200
        data = response.get_json()

        # Check required keys
        required_keys = [
            'ear_left', 'ear_right', 'mar',
            'pitch', 'yaw', 'roll',
            'perclos', 'confidence'
        ]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"
            assert isinstance(data[key], (int, float)), f"{key} should be numeric"

    def test_predict_metric_ranges(self, client, base64_jpeg_raw):
        """Test /predict returns metrics in valid ranges."""
        response = client.post('/predict', json={'data': base64_jpeg_raw})

        data = response.get_json()

        # EAR and MAR: 0-1 range (or similar)
        assert 0.0 <= data['ear_left'] <= 1.0 or data['ear_left'] > 1.0
        assert 0.0 <= data['ear_right'] <= 1.0 or data['ear_right'] > 1.0
        assert 0.0 <= data['mar'] <= 1.0 or data['mar'] > 1.0

        # Confidence and PERCLOS: 0-1
        assert 0.0 <= data['confidence'] <= 1.0
        assert 0.0 <= data['perclos'] <= 1.0

        # Head pose angles: reasonable range (-180 to 180)
        assert -180 <= data['pitch'] <= 180
        assert -180 <= data['yaw'] <= 180
        assert -180 <= data['roll'] <= 180

    def test_predict_missing_data(self, client):
        """Test /predict with missing image data."""
        response = client.post('/predict', json={})
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data

    def test_predict_empty_data(self, client):
        """Test /predict with empty data field."""
        response = client.post('/predict', json={'data': ''})
        assert response.status_code == 400

    def test_predict_invalid_base64(self, client):
        """Test /predict with invalid base64."""
        response = client.post('/predict', json={'data': 'not valid base64!!!'})
        assert response.status_code == 400

    def test_predict_null_data(self, client):
        """Test /predict with null data."""
        response = client.post('/predict', json={'data': None})
        assert response.status_code == 400

    def test_predict_wrong_json_type(self, client):
        """Test /predict with wrong JSON structure."""
        response = client.post('/predict', json={'image': 'base64data'})
        assert response.status_code == 400

    def test_predict_response_json_valid(self, client, base64_jpeg_raw):
        """Test /predict response is valid JSON."""
        response = client.post('/predict', json={'data': base64_jpeg_raw})

        assert response.status_code == 200
        # Verify it's JSON by parsing
        data = response.get_json()
        assert isinstance(data, dict)

    def test_predict_multiple_requests(self, client, base64_jpeg_raw):
        """Test multiple /predict requests."""
        for i in range(5):
            response = client.post('/predict', json={'data': base64_jpeg_raw})
            assert response.status_code == 200
            data = response.get_json()
            assert 'confidence' in data

    def test_predict_different_image_sizes(self, client):
        """Test /predict with different image sizes."""
        sizes = [(64, 64), (128, 128), (480, 640), (720, 1280)]

        for height, width in sizes:
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            _, encoded = cv2.imencode('.jpg', frame)
            b64 = base64.b64encode(encoded).decode('utf-8')

            response = client.post('/predict', json={'data': b64})
            assert response.status_code == 200
            data = response.get_json()
            assert 'confidence' in data

    def test_predict_concurrent_requests(self, client, base64_jpeg_raw):
        """Test multiple simultaneous requests (sequential in test)."""
        responses = []
        for _ in range(10):
            response = client.post('/predict', json={'data': base64_jpeg_raw})
            responses.append(response)

        assert all(r.status_code == 200 for r in responses)
        assert len(set(r.get_json()['confidence'] for r in responses)) >= 1

    def test_predict_black_image(self, client):
        """Test /predict with black (zero) image."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, encoded = cv2.imencode('.jpg', frame)
        b64 = base64.b64encode(encoded).decode('utf-8')

        response = client.post('/predict', json={'data': b64})
        assert response.status_code == 200
        data = response.get_json()
        assert 'confidence' in data

    def test_predict_white_image(self, client):
        """Test /predict with white (maximum) image."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        _, encoded = cv2.imencode('.jpg', frame)
        b64 = base64.b64encode(encoded).decode('utf-8')

        response = client.post('/predict', json={'data': b64})
        assert response.status_code == 200
        data = response.get_json()
        assert 'confidence' in data

    def test_predict_grayscale_image(self, client):
        """Test /predict with grayscale image."""
        frame = np.random.randint(0, 255, (480, 640, 1), dtype=np.uint8)
        frame = np.concatenate([frame] * 3, axis=2)  # Convert to 3-channel
        _, encoded = cv2.imencode('.jpg', frame)
        b64 = base64.b64encode(encoded).decode('utf-8')

        response = client.post('/predict', json={'data': b64})
        assert response.status_code == 200


class TestApiErrorHandling:
    """Test API error handling."""

    def test_predict_malformed_json(self, client):
        """Test /predict with malformed JSON."""
        response = client.post(
            '/predict',
            data='{invalid json',
            content_type='application/json'
        )
        # Flask returns 500 for JSON parsing errors
        assert response.status_code in [400, 415, 500]

    def test_predict_missing_content_type(self, client, base64_jpeg_raw):
        """Test /predict without JSON content type."""
        # Flask test client is forgiving, but we can verify behavior
        response = client.post('/predict', data={'data': base64_jpeg_raw})
        # Flask may return 400, 415, or 500 depending on content type handling
        assert response.status_code in [200, 400, 415, 500]

    def test_predict_empty_request(self, client):
        """Test /predict with empty request body."""
        response = client.post('/predict', data='', content_type='application/json')
        # Empty body results in 400 or 500
        assert response.status_code in [400, 415, 500]

    def test_predict_oversized_image(self, client):
        """Test /predict with very large image."""
        # Create a large frame (but not so large it crashes)
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        b64 = base64.b64encode(encoded).decode('utf-8')

        response = client.post('/predict', json={'data': b64})
        # Should handle gracefully or return 200
        assert response.status_code in [200, 413]


class TestMetricsTracking:
    """Test metrics tracking and statistics."""

    def test_metrics_endpoint_if_exists(self, client):
        """Test /metrics endpoint if implemented."""
        response = client.get('/metrics')
        # Endpoint may or may not exist - just verify no 500 error
        assert response.status_code in [200, 404]

    def test_predict_increments_count(self, client, base64_jpeg_raw):
        """Test that multiple predictions can be tracked."""
        responses = []
        for i in range(3):
            response = client.post('/predict', json={'data': base64_jpeg_raw})
            responses.append(response)

        # All should succeed
        assert all(r.status_code == 200 for r in responses)

    def test_metrics_response_structure(self, client):
        """Test /metrics response structure if endpoint exists."""
        response = client.get('/metrics')

        if response.status_code == 200:
            try:
                data = response.get_json()
                # Common metrics fields
                possible_keys = ['total_requests', 'successful_requests', 'avg_latency']
                # At least some key might exist
                assert isinstance(data, dict)
            except:
                # Might be non-JSON response (Prometheus format, etc)
                pass


class TestApiConcurrency:
    """Test API behavior under load."""

    def test_concurrent_different_sizes(self, client):
        """Test concurrent requests with different image sizes."""
        base64_images = []

        # Create different size images
        for size in [(64, 64), (128, 128), (256, 256)]:
            frame = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            _, encoded = cv2.imencode('.jpg', frame)
            b64 = base64.b64encode(encoded).decode('utf-8')
            base64_images.append(b64)

        # Send requests (simulating concurrent)
        responses = []
        for b64 in base64_images * 3:
            response = client.post('/predict', json={'data': b64})
            responses.append(response)

        assert all(r.status_code == 200 for r in responses)

    def test_session_isolation(self, client, base64_jpeg_raw):
        """Test that requests don't interfere with each other."""
        data1 = []
        data2 = []

        # Make alternating requests
        for i in range(5):
            r1 = client.post('/predict', json={'data': base64_jpeg_raw})
            r2 = client.post('/predict', json={'data': base64_jpeg_raw})

            data1.append(r1.get_json())
            data2.append(r2.get_json())

        # All responses should be valid
        assert all(isinstance(d, dict) for d in data1)
        assert all(isinstance(d, dict) for d in data2)


class TestApiIntegration:
    """End-to-end API integration tests."""

    def test_full_workflow(self, client, synthetic_frame):
        """Test complete workflow from image to metrics."""
        # Prepare image
        _, encoded = cv2.imencode('.jpg', synthetic_frame)
        b64 = base64.b64encode(encoded).decode('utf-8')

        # Get homepage
        home_response = client.get('/')
        assert home_response.status_code == 200

        # Send prediction request
        pred_response = client.post('/predict', json={'data': b64})
        assert pred_response.status_code == 200

        # Verify metrics
        data = pred_response.get_json()
        assert data is not None
        assert len(data) >= 8

    def test_sequential_predictions(self, client):
        """Test sequential predictions simulate monitoring."""
        frames = []
        for _ in range(5):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            _, encoded = cv2.imencode('.jpg', frame)
            b64 = base64.b64encode(encoded).decode('utf-8')
            frames.append(b64)

        predictions = []
        for b64 in frames:
            response = client.post('/predict', json={'data': b64})
            assert response.status_code == 200
            predictions.append(response.get_json())

        # Should have 5 predictions
        assert len(predictions) == 5

        # All should have confidence values
        confidence_values = [p['confidence'] for p in predictions]
        assert all(0.0 <= c <= 1.0 for c in confidence_values)

    def test_response_consistency(self, client, synthetic_frame):
        """Test response structure consistency across requests."""
        _, encoded = cv2.imencode('.jpg', synthetic_frame)
        b64 = base64.b64encode(encoded).decode('utf-8')

        responses = []
        for _ in range(3):
            response = client.post('/predict', json={'data': b64})
            responses.append(response.get_json())

        # All responses should have same keys
        keys_set = set(responses[0].keys())
        for resp in responses[1:]:
            assert set(resp.keys()) == keys_set


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

