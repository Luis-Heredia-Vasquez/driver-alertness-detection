"""Dataset loading and TF pipeline utilities.

This module implements dataset parsers for MRL Eye Dataset and NTHU-DDD
folder structures, simple augmentations (brightness jitter, horizontal flip,
Gaussian noise), frame sampling utilities, and helpers to build
`tf.data.Dataset` pipelines for train/val/test splits.

Note: TensorFlow is required to build the `tf.data.Dataset` pipelines.
"""
from typing import Iterable, List, Tuple, Callable, Optional
import os
import random

import numpy as np

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - import-time
    tf = None


def list_mrl_dataset(root: str) -> List[Tuple[str, int]]:
    """Walk a simple MRL Eye dataset structure and return list of (frame_path, label).

    Expected layout (example):
        root/
          subject1/
            open/
              img001.jpg
            closed/
              img001.jpg

    Args:
        root: dataset root directory

    Returns:
        list of (path, label) where label is 0/1 (0=open, 1=closed) — heuristic.
    """
    items = []
    for subject in sorted(os.listdir(root)):
        subj_path = os.path.join(root, subject)
        if not os.path.isdir(subj_path):
            continue
        for state in os.listdir(subj_path):
            state_path = os.path.join(subj_path, state)
            if not os.path.isdir(state_path):
                continue
            label = 1 if 'closed' in state.lower() else 0
            for fname in sorted(os.listdir(state_path)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    items.append((os.path.join(state_path, fname), label))
    return items


def list_nthu_ddd(root: str) -> List[Tuple[str, int]]:
    """Parse NTHU-DDD-like structure and return list of (frame_path, label).

    This is a heuristic parser that expects folders per video and optional
    metadata files; for unit tests, only the path walking logic matters.
    """
    items = []
    for sub in sorted(os.listdir(root)):
        subp = os.path.join(root, sub)
        if not os.path.isdir(subp):
            continue
        for fname in sorted(os.listdir(subp)):
            if fname.lower().endswith(('.jpg', '.png')):
                # label unknown in this generic parser; default 0
                items.append((os.path.join(subp, fname), 0))
    return items


def brightness_jitter(image: np.ndarray, factor: float) -> np.ndarray:
    """Adjust brightness by a factor. Image in [0,255]."""
    img = image.astype(np.float32) * factor
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def horizontal_flip(image: np.ndarray) -> np.ndarray:
    """Return horizontally flipped image."""
    return np.ascontiguousarray(np.fliplr(image))


def gaussian_noise(image: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """Add Gaussian noise (stddev=sigma) to image; returns uint8."""
    noise = np.random.normal(0, sigma, image.shape)
    img = image.astype(np.float32) + noise
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def sample_frames_uniform(frame_paths: List[str], num_samples: int) -> List[str]:
    """Uniformly sample `num_samples` frame paths from `frame_paths`.

    If there are fewer frames than requested, frames are repeated.
    """
    if len(frame_paths) == 0:
        return []
    if len(frame_paths) >= num_samples:
        indices = np.linspace(0, len(frame_paths) - 1, num=num_samples, dtype=int)
        return [frame_paths[i] for i in indices]
    # repeat frames
    reps = int(np.ceil(num_samples / len(frame_paths)))
    extended = frame_paths * reps
    return extended[:num_samples]


def build_tf_dataset(
    samples: Iterable[Tuple[str, int]],
    batch_size: int = 8,
    image_size: Tuple[int, int] = (64, 64),
    shuffle: bool = True,
    augment: bool = False,
) -> 'tf.data.Dataset':
    """Build a `tf.data.Dataset` from (path, label) samples.

    Args:
        samples: iterable of (image_path, label)
        batch_size: batch size
        image_size: (w,h) target size
        shuffle: whether to shuffle
        augment: whether to apply simple augmentations using numpy ops

    Returns:
        tf.data.Dataset yielding (image_batch, label_batch)
    """
    if tf is None:
        raise RuntimeError("TensorFlow is required to build tf.data.Dataset")

    paths = [p for p, _ in samples]
    labels = [int(l) for _, l in samples]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths))

    def _load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, image_size)
        # optional numpy-based augmentation via tf.numpy_function
        if augment:
            def _aug(x):
                x = (x * 255).astype('uint8')
                # random brightness
                if random.random() < 0.5:
                    factor = random.uniform(0.7, 1.3)
                    x = brightness_jitter(x, factor)
                # random flip
                if random.random() < 0.5:
                    x = horizontal_flip(x)
                # gaussian noise
                if random.random() < 0.5:
                    x = gaussian_noise(x, sigma=random.uniform(1.0, 8.0))
                x = x.astype('float32') / 255.0
                return x

            img = tf.numpy_function(_aug, [img], tf.float32)
            img.set_shape((*image_size, 3))

        return img, label

    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
