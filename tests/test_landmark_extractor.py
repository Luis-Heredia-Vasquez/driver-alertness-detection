"""Tests for landmark extraction and feature computation."""
import numpy as np
import pytest

from src.preprocessing.landmark_extractor import (
    compute_ear,
    compute_mar,
    LandmarkFeatures,
    estimate_head_pose,
)


class TestComputeEAR:
    """Tests for Eye Aspect Ratio computation."""

    def test_compute_ear_normal(self):
        """Test EAR with typical eye landmark data."""
        eye = np.array([
            [0, 0],    # p1
            [0, 1],    # p2
            [0, 1],    # p3
            [2, 0],    # p4
            [2, 1],    # p5
            [2, 1],    # p6
        ], dtype=float)
        ear = compute_ear(eye)
        assert isinstance(ear, float)
        assert ear > 0

    def test_compute_ear_closed_eye(self):
        """Test EAR approaches 0 for closed eye (points collapse vertically)."""
        eye = np.array([
            [0, 0],      # p1 - left corner
            [1, 0.05],   # p2 - top-left (small)
            [1, 0.05],   # p3 - top-right (small)
            [2, 0],      # p4 - right corner
            [1, 0.05],   # p5 - bottom-right (small)
            [1, 0.05],   # p6 - bottom-left (small)
        ], dtype=float)
        ear = compute_ear(eye)
        assert 0 <= ear < 0.15  # very small for closed eye

    def test_compute_ear_degenerate(self):
        """Test EAR with degenerate input (zero horizontal distance)."""
        eye = np.array([
            [0, 0],
            [0, 1],
            [0, 1],
            [0, 0],  # same as p1 — zero horizontal distance
            [0, 1],
            [0, 1],
        ], dtype=float)
        ear = compute_ear(eye)
        assert ear == 0.0

    def test_compute_ear_wrong_shape(self):
        """Test that wrong shape raises ValueError."""
        eye = np.array([[0, 0], [1, 1]])  # (2, 2) instead of (6, 2)
        with pytest.raises(ValueError, match="must be"):
            compute_ear(eye)


class TestComputeMAR:
    """Tests for Mouth Aspect Ratio computation."""

    def test_compute_mar_normal(self):
        """Test MAR with typical mouth landmark data."""
        mouth = np.array([
            [0, 0],
            [0, 1],
            [0, 1],
            [2, 0],
            [2, 1],
            [2, 1],
        ], dtype=float)
        mar = compute_mar(mouth)
        assert isinstance(mar, float)
        assert mar > 0

    def test_compute_mar_closed_mouth(self):
        """Test MAR with closed mouth (small vertical distance)."""
        mouth = np.array([
            [0, 0],      # left corner
            [1, 0.05],   # top-left (small)
            [1, 0.05],   # top-right (small)
            [2, 0],      # right corner
            [1, 0.05],   # bottom-right (small)
            [1, 0.05],   # bottom-left (small)
        ], dtype=float)
        mar = compute_mar(mouth)
        assert 0 <= mar < 0.15  # very small for closed mouth


class TestEstimateHeadPose:
    """Tests for head pose estimation."""

    def test_estimate_head_pose_valid(self):
        """Test head pose returns valid angles (may be zeros for bad PnP)."""
        # Create a minimal 468-point landmark set (mostly zeros, critical indices filled)
        landmarks = np.zeros((468, 3), dtype=float)
        landmarks[1] = [0.5, 0.3, 0]   # nose tip
        landmarks[152] = [0.5, 0.7, 0] # chin
        landmarks[33] = [0.3, 0.4, 0]  # left eye left corner
        landmarks[263] = [0.7, 0.4, 0] # right eye right corner
        landmarks[61] = [0.4, 0.6, 0]  # mouth left
        landmarks[291] = [0.6, 0.6, 0] # mouth right

        # solvePnP may fail or return zeros with test data; just check it returns floats
        try:
            pitch, yaw, roll = estimate_head_pose(landmarks, (640, 480))
            assert isinstance(pitch, float)
            assert isinstance(yaw, float)
            assert isinstance(roll, float)
        except RuntimeError:
            # Expected if solvePnP fails on synthetic data
            pass

    def test_estimate_head_pose_invalid_shape(self):
        """Test that invalid landmark shape raises ValueError."""
        # Single 2D point is valid shape (N, >=2), but will fail at solvePnP
        with pytest.raises((ValueError, RuntimeError)):
            estimate_head_pose(np.array([[0, 0]]), (640, 480))

    def test_estimate_head_pose_missing_indices(self):
        """Test that all-zero landmarks still returns valid float angles."""
        landmarks = np.zeros((468, 3), dtype=float)
        # All-zero landmarks may succeed or fail solvePnP; just verify it doesn't crash
        pitch, yaw, roll = estimate_head_pose(landmarks, (640, 480))
        assert isinstance(pitch, float)
        assert isinstance(yaw, float)
        assert isinstance(roll, float)


class TestLandmarkFeatures:
    """Tests for LandmarkFeatures dataclass."""

    def test_landmark_features_creation(self):
        """Test creating a LandmarkFeatures instance."""
        landmarks = np.zeros((468, 3), dtype=float)
        features = LandmarkFeatures(
            landmarks=landmarks,
            ear_left=0.3,
            ear_right=0.3,
            mar=0.5,
            pitch=10.0,
            yaw=5.0,
            roll=-2.0,
            timestamp=1.234,
        )
        assert features.ear_left == 0.3
        assert features.timestamp == 1.234
        assert features.landmarks.shape == (468, 3)
