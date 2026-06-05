"""Feature engineering helpers: sliding window aggregation and blink/yawn detection.

Provides pure, unit-testable functions which operate on numerical series such as
EAR and MAR to compute summary statistics per sliding window (mean, std,
PERCLOS, blink rate, yawn detection).
"""
from typing import List, Tuple
import numpy as np


def sliding_windows(array: np.ndarray, window_size: int, step: int) -> List[np.ndarray]:
    """Yield sliding windows from a 1D numpy array.

    Args:
        array: 1D array
        window_size: number of frames per window
        step: step size between windows

    Returns:
        list of windows (numpy arrays)
    """
    arr = np.asarray(array)
    if arr.ndim != 1:
        raise ValueError("array must be 1D")
    if window_size <= 0 or step <= 0:
        raise ValueError("window_size and step must be > 0")
    windows = []
    for start in range(0, len(arr) - window_size + 1, step):
        windows.append(arr[start:start + window_size])
    return windows


def perclos(ear_window: np.ndarray, threshold: float) -> float:
    """Compute PERCLOS: percent of frames where EAR is below threshold.

    Args:
        ear_window: 1D array of EAR values for the window
        threshold: EAR threshold for eyes-closed

    Returns:
        fraction in [0,1]
    """
    ear_window = np.asarray(ear_window)
    if ear_window.size == 0:
        return 0.0
    below = np.sum(ear_window < threshold)
    return float(below / ear_window.size)


def count_blinks(ear_window: np.ndarray, threshold: float, min_frames: int = 1) -> int:
    """Count blinks in an EAR series using threshold crossings.

    A blink is counted when EAR falls below `threshold` and then rises above it.
    `min_frames` can be used to suppress very short fluctuations.
    """
    ear = np.asarray(ear_window)
    if ear.size == 0:
        return 0
    closed = ear < threshold
    blinks = 0
    i = 0
    while i < len(closed):
        if closed[i]:
            # count this closed segment
            j = i
            while j < len(closed) and closed[j]:
                j += 1
            if (j - i) >= min_frames:
                blinks += 1
            i = j
        else:
            i += 1
    return int(blinks)


def detect_yawn(mar_window: np.ndarray, threshold: float, min_frames: int = 3) -> bool:
    """Detect a yawn if MAR is above threshold for at least min_frames frames."""
    mar = np.asarray(mar_window)
    if mar.size == 0:
        return False
    above = mar > threshold
    # find any run of consecutive True of length >= min_frames
    count = 0
    for v in above:
        if v:
            count += 1
            if count >= min_frames:
                return True
        else:
            count = 0
    return False


def aggregate_window_features(
    ear_series: List[float],
    mar_series: List[float],
    window_size: int,
    step: int,
    ear_threshold: float = 0.2,
    mar_threshold: float = 0.6,
) -> List[dict]:
    """Compute aggregated features for sliding windows over EAR and MAR.

    Returns a list of dicts with keys: mean_ear, std_ear, perclos, blinks, yawn
    """
    ear_arr = np.asarray(ear_series)
    mar_arr = np.asarray(mar_series)
    if ear_arr.shape[0] != mar_arr.shape[0]:
        raise ValueError("EAR and MAR series must have same length")

    windows = sliding_windows(ear_arr, window_size, step)
    features = []
    for i, w in enumerate(windows):
        start = i * step
        mar_w = mar_arr[start:start + window_size]
        mean_ear = float(np.mean(w)) if w.size else 0.0
        std_ear = float(np.std(w)) if w.size else 0.0
        p = perclos(w, ear_threshold)
        blinks = count_blinks(w, ear_threshold)
        yawn = detect_yawn(mar_w, mar_threshold)
        features.append({
            'mean_ear': mean_ear,
            'std_ear': std_ear,
            'perclos': p,
            'blinks': blinks,
            'yawn': yawn,
        })
    return features
