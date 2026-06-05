"""Pytest fixtures and environment mocks to avoid heavy dependencies.

This module injects lightweight mocks for `tensorflow` and `mediapipe` into
`sys.modules` so tests can import modules that reference them without
installing the actual packages. Also provides reusable test fixtures.
"""
import sys
import types
import pytest
import numpy as np
import base64
import cv2
from pathlib import Path


def _make_dummy_module(name: str):
    m = types.ModuleType(name)
    return m


def pytest_configure():
    # Insert dummy tensorflow if not installed
    if 'tensorflow' not in sys.modules:
        sys.modules['tensorflow'] = _make_dummy_module('tensorflow')
        # minimal attributes used by code
        sys.modules['tensorflow'].data = types.SimpleNamespace()

    # Insert dummy mediapipe if not installed
    if 'mediapipe' not in sys.modules:
        mp = _make_dummy_module('mediapipe')
        # create nested mp.solutions.face_mesh with FaceMesh dummy class
        sol = types.SimpleNamespace()
        fm = types.SimpleNamespace()

        class _DummyFaceMesh:
            def __init__(self, *args, **kwargs):
                pass

            def process(self, image):
                return types.SimpleNamespace(multi_face_landmarks=None)

        fm.FaceMesh = _DummyFaceMesh
        sol.face_mesh = fm
        mp.solutions = sol
        sys.modules['mediapipe'] = mp


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_landmarks():
    """Create 468 synthetic facial landmarks."""
    return np.random.randn(468, 3).astype(np.float32)


@pytest.fixture
def sample_eye_points():
    """Create 6 synthetic eye landmark points (for EAR calculation)."""
    return np.array([
        [0.1, 0.1],    # P1
        [0.15, 0.05],  # P2
        [0.2, 0.1],    # P3
        [0.2, 0.2],    # P4
        [0.15, 0.25],  # P5
        [0.1, 0.2]     # P6
    ], dtype=np.float32)


@pytest.fixture
def sample_mouth_points():
    """Create 6 synthetic mouth landmark points (for MAR calculation)."""
    return np.array([
        [0.3, 0.3],    # Top lip
        [0.35, 0.35],  # Top left
        [0.4, 0.35],   # Top right
        [0.4, 0.5],    # Bottom right
        [0.35, 0.55],  # Bottom
        [0.3, 0.5]     # Bottom left
    ], dtype=np.float32)


@pytest.fixture
def synthetic_frame():
    """Create a synthetic BGR video frame (480x640x3)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add some random noise/patterns
    frame[100:200, 100:200] = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    frame[300:400, 300:400] = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return frame


@pytest.fixture
def base64_jpeg_image(synthetic_frame):
    """Convert synthetic frame to base64-encoded JPEG data URL."""
    _, encoded = cv2.imencode('.jpg', synthetic_frame)
    b64 = base64.b64encode(encoded).decode('utf-8')
    return f'data:image/jpeg;base64,{b64}'


@pytest.fixture
def base64_jpeg_raw(synthetic_frame):
    """Convert synthetic frame to base64-encoded JPEG (raw, no data URL)."""
    _, encoded = cv2.imencode('.jpg', synthetic_frame)
    return base64.b64encode(encoded).decode('utf-8')


@pytest.fixture
def flask_client():
    """Create Flask test client for integration tests."""
    from src.api.app import create_app
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_ear_series():
    """Generate 100 sample EAR values (eye aspect ratio)."""
    return np.random.uniform(0.15, 0.35, 100).astype(np.float32)


@pytest.fixture
def sample_mar_series():
    """Generate 100 sample MAR values (mouth aspect ratio)."""
    return np.random.uniform(0.4, 0.8, 100).astype(np.float32)


@pytest.fixture
def closed_eye_series():
    """Generate EAR series with many low values (closed eyes)."""
    # 30% of frames below 0.2 (eyes closed), rest open
    series = np.random.uniform(0.25, 0.35, 70)  # Open
    closed = np.random.uniform(0.05, 0.15, 30)  # Closed
    combined = np.concatenate([series, closed])
    np.random.shuffle(combined)
    return combined.astype(np.float32)
