"""Tests for data pipeline utilities."""
import tempfile
import os
import pytest
import numpy as np

from src.preprocessing.data_pipeline import (
    list_mrl_dataset,
    list_nthu_ddd,
    brightness_jitter,
    horizontal_flip,
    gaussian_noise,
    sample_frames_uniform,
)


class TestBrightnessJitter:
    """Tests for brightness_jitter augmentation."""

    def test_brightness_jitter_brighten(self):
        """Test brightening an image."""
        img = np.ones((10, 10, 3), dtype=np.uint8) * 100
        result = brightness_jitter(img, factor=1.5)
        assert result.dtype == np.uint8
        assert np.all(result >= 100)  # brightened

    def test_brightness_jitter_darken(self):
        """Test darkening an image."""
        img = np.ones((10, 10, 3), dtype=np.uint8) * 100
        result = brightness_jitter(img, factor=0.5)
        assert result.dtype == np.uint8
        assert np.all(result <= 100)  # darkened

    def test_brightness_jitter_clipping(self):
        """Test that values are clipped to [0, 255]."""
        img = np.ones((10, 10, 3), dtype=np.uint8) * 200
        result = brightness_jitter(img, factor=2.0)
        assert np.all(result <= 255)
        assert result.dtype == np.uint8


class TestHorizontalFlip:
    """Tests for horizontal_flip augmentation."""

    def test_horizontal_flip_simple(self):
        """Test basic horizontal flip."""
        img = np.array([
            [[1, 0, 0], [2, 0, 0]],
            [[3, 0, 0], [4, 0, 0]],
        ], dtype=np.uint8)
        result = horizontal_flip(img)
        expected = np.array([
            [[2, 0, 0], [1, 0, 0]],
            [[4, 0, 0], [3, 0, 0]],
        ], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)


class TestGaussianNoise:
    """Tests for gaussian_noise augmentation."""

    def test_gaussian_noise_output_shape(self):
        """Test that output shape matches input."""
        img = np.ones((10, 10, 3), dtype=np.uint8) * 128
        result = gaussian_noise(img, sigma=5.0)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_gaussian_noise_clipping(self):
        """Test that output is clipped to [0, 255]."""
        img = np.ones((20, 20, 3), dtype=np.uint8) * 200
        result = gaussian_noise(img, sigma=50.0)
        assert np.all(result >= 0)
        assert np.all(result <= 255)


class TestSampleFramesUniform:
    """Tests for uniform frame sampling."""

    def test_sample_frames_exact(self):
        """Test sampling exact count from sufficient frames."""
        frames = [f"frame_{i}.jpg" for i in range(10)]
        result = sample_frames_uniform(frames, num_samples=5)
        assert len(result) == 5

    def test_sample_frames_repeat(self):
        """Test sampling repeats when fewer frames than requested."""
        frames = [f"frame_{i}.jpg" for i in range(3)]
        result = sample_frames_uniform(frames, num_samples=10)
        assert len(result) == 10

    def test_sample_frames_empty(self):
        """Test sampling from empty list."""
        result = sample_frames_uniform([], num_samples=5)
        assert len(result) == 0


class TestMRLDataset:
    """Tests for MRL Eye Dataset parser."""

    def test_list_mrl_dataset_simple(self):
        """Test parsing a simple MRL-like directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create structure:
            # subject1/open/img001.jpg
            # subject1/closed/img002.jpg
            subj_dir = os.path.join(tmpdir, "subject1")
            os.makedirs(os.path.join(subj_dir, "open"), exist_ok=True)
            os.makedirs(os.path.join(subj_dir, "closed"), exist_ok=True)
            
            open_path = os.path.join(subj_dir, "open", "img001.jpg")
            closed_path = os.path.join(subj_dir, "closed", "img002.jpg")
            open(open_path, 'w').close()
            open(closed_path, 'w').close()

            items = list_mrl_dataset(tmpdir)
            assert len(items) == 2
            paths = [p for p, _ in items]
            labels = [l for _, l in items]
            assert any("open" in p for p in paths)
            assert any("closed" in p for p in paths)
            # label=0 for open, 1 for closed
            assert 0 in labels
            assert 1 in labels

    def test_list_mrl_dataset_empty(self):
        """Test parsing empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            items = list_mrl_dataset(tmpdir)
            assert len(items) == 0


class TestNTHUDDD:
    """Tests for NTHU-DDD dataset parser."""

    def test_list_nthu_ddd_simple(self):
        """Test parsing a simple NTHU-DDD-like structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create structure:
            # video1/frame001.jpg
            # video1/frame002.jpg
            vid_dir = os.path.join(tmpdir, "video1")
            os.makedirs(vid_dir, exist_ok=True)
            
            for i in range(1, 3):
                fpath = os.path.join(vid_dir, f"frame{i:03d}.jpg")
                open(fpath, 'w').close()

            items = list_nthu_ddd(tmpdir)
            assert len(items) == 2
            assert all(l == 0 for _, l in items)  # default label 0

    def test_list_nthu_ddd_empty(self):
        """Test parsing empty NTHU-DDD directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            items = list_nthu_ddd(tmpdir)
            assert len(items) == 0
