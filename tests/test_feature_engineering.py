"""Tests for feature engineering and sliding window aggregation."""
import numpy as np
import pytest

from src.preprocessing.feature_engineering import (
    sliding_windows,
    perclos,
    count_blinks,
    detect_yawn,
    aggregate_window_features,
)


class TestSlidingWindows:
    """Tests for sliding window extraction."""

    def test_sliding_windows_basic(self):
        """Test basic sliding window extraction."""
        arr = np.array([1, 2, 3, 4, 5], dtype=float)
        windows = sliding_windows(arr, window_size=2, step=1)
        assert len(windows) == 4
        np.testing.assert_array_equal(windows[0], [1, 2])
        np.testing.assert_array_equal(windows[-1], [4, 5])

    def test_sliding_windows_with_step(self):
        """Test sliding windows with step > 1."""
        arr = np.array([1, 2, 3, 4, 5, 6], dtype=float)
        windows = sliding_windows(arr, window_size=2, step=2)
        assert len(windows) == 3
        np.testing.assert_array_equal(windows[0], [1, 2])
        np.testing.assert_array_equal(windows[1], [3, 4])

    def test_sliding_windows_insufficient_frames(self):
        """Test with fewer frames than window size."""
        arr = np.array([1, 2], dtype=float)
        windows = sliding_windows(arr, window_size=5, step=1)
        assert len(windows) == 0

    def test_sliding_windows_invalid_input(self):
        """Test with invalid inputs."""
        with pytest.raises(ValueError):
            sliding_windows(np.array([[1, 2]]), window_size=2, step=1)  # 2D array

        with pytest.raises(ValueError):
            sliding_windows(np.array([1, 2]), window_size=0, step=1)  # invalid window_size


class TestPerclos:
    """Tests for PERCLOS (% frames with closed eyes)."""

    def test_perclos_all_open(self):
        """Test PERCLOS when all eyes are open."""
        ear = np.array([0.3, 0.35, 0.32, 0.31], dtype=float)
        p = perclos(ear, threshold=0.2)
        assert p == 0.0  # all above threshold (0% closed)

    def test_perclos_all_closed(self):
        """Test PERCLOS when all eyes are closed."""
        ear = np.array([0.1, 0.15, 0.12], dtype=float)
        p = perclos(ear, threshold=0.2)
        assert p == 1.0  # all below threshold

    def test_perclos_mixed(self):
        """Test PERCLOS with mixed open/closed."""
        ear = np.array([0.3, 0.1, 0.35, 0.15], dtype=float)
        p = perclos(ear, threshold=0.2)
        assert p == 0.5  # 2 out of 4 below threshold

    def test_perclos_empty(self):
        """Test PERCLOS with empty array."""
        p = perclos(np.array([]), threshold=0.2)
        assert p == 0.0


class TestCountBlinks:
    """Tests for blink counting."""

    def test_count_blinks_single(self):
        """Test counting a single blink."""
        ear = np.array([0.3, 0.3, 0.1, 0.1, 0.3, 0.3], dtype=float)
        blinks = count_blinks(ear, threshold=0.2)
        assert blinks == 1

    def test_count_blinks_multiple(self):
        """Test counting multiple blinks."""
        ear = np.array([0.3, 0.1, 0.3, 0.1, 0.3], dtype=float)
        blinks = count_blinks(ear, threshold=0.2)
        assert blinks == 2

    def test_count_blinks_none(self):
        """Test counting with no blinks."""
        ear = np.array([0.3, 0.35, 0.32, 0.31], dtype=float)
        blinks = count_blinks(ear, threshold=0.2)
        assert blinks == 0

    def test_count_blinks_min_frames(self):
        """Test min_frames threshold for blink detection."""
        ear = np.array([0.3, 0.1, 0.3, 0.1, 0.1, 0.3], dtype=float)
        # with min_frames=1: count all dips
        blinks_1 = count_blinks(ear, threshold=0.2, min_frames=1)
        # with min_frames=2: only count dips >= 2 frames
        blinks_2 = count_blinks(ear, threshold=0.2, min_frames=2)
        assert blinks_1 >= blinks_2


class TestDetectYawn:
    """Tests for yawn detection."""

    def test_detect_yawn_true(self):
        """Test detecting a yawn (MAR above threshold)."""
        mar = np.array([0.5, 0.65, 0.7, 0.68, 0.5], dtype=float)
        is_yawn = detect_yawn(mar, threshold=0.6, min_frames=3)
        assert is_yawn is True

    def test_detect_yawn_false(self):
        """Test not detecting a yawn (MAR below threshold)."""
        mar = np.array([0.5, 0.55, 0.52, 0.54, 0.51], dtype=float)
        is_yawn = detect_yawn(mar, threshold=0.6, min_frames=3)
        assert is_yawn is False

    def test_detect_yawn_short_burst(self):
        """Test that short bursts above threshold aren't counted."""
        mar = np.array([0.5, 0.65, 0.5, 0.5, 0.5], dtype=float)
        is_yawn = detect_yawn(mar, threshold=0.6, min_frames=3)
        assert is_yawn is False

    def test_detect_yawn_empty(self):
        """Test with empty array."""
        is_yawn = detect_yawn(np.array([]), threshold=0.6)
        assert is_yawn is False


class TestAggregateWindowFeatures:
    """Tests for sliding window feature aggregation."""

    def test_aggregate_window_features_basic(self):
        """Test basic feature aggregation over windows."""
        ear_series = [0.3, 0.35, 0.1, 0.15, 0.3, 0.32]
        mar_series = [0.5, 0.55, 0.52, 0.68, 0.7, 0.65]
        
        features = aggregate_window_features(
            ear_series,
            mar_series,
            window_size=2,
            step=1,
            ear_threshold=0.2,
            mar_threshold=0.6,
        )
        
        assert len(features) == 5  # 6 frames, window=2, step=1 -> 5 windows
        for feat in features:
            assert 'mean_ear' in feat
            assert 'std_ear' in feat
            assert 'perclos' in feat
            assert 'blinks' in feat
            assert 'yawn' in feat

    def test_aggregate_window_features_mismatched_length(self):
        """Test that mismatched series lengths raise ValueError."""
        ear_series = [0.3, 0.35]
        mar_series = [0.5, 0.55, 0.52]  # different length
        
        with pytest.raises(ValueError, match="same length"):
            aggregate_window_features(
                ear_series,
                mar_series,
                window_size=2,
                step=1,
            )

    def test_aggregate_window_features_empty(self):
        """Test with empty series."""
        features = aggregate_window_features(
            [],
            [],
            window_size=2,
            step=1,
        )
        assert len(features) == 0

    def test_aggregate_window_features_values(self):
        """Test that aggregated values are reasonable."""
        ear_series = [0.3, 0.32, 0.1, 0.15]
        mar_series = [0.5, 0.55, 0.65, 0.7]
        
        features = aggregate_window_features(
            ear_series,
            mar_series,
            window_size=2,
            step=1,
            ear_threshold=0.2,
            mar_threshold=0.6,
        )
        
        # first window: EAR=[0.3, 0.32], MAR=[0.5, 0.55]
        assert features[0]['mean_ear'] == pytest.approx(0.31, abs=0.01)
        assert 0 <= features[0]['perclos'] <= 1
        assert features[0]['blinks'] >= 0
        assert isinstance(features[0]['yawn'], bool)
