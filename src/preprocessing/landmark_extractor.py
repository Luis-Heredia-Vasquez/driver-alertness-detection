"""MediaPipe-based landmark extraction and feature computation.

This module provides utilities to extract facial landmarks using MediaPipe
FaceMesh and compute common metrics used for driver alertness detection:
- Eye Aspect Ratio (EAR)
- Mouth Aspect Ratio (MAR)
- Head pose estimation (pitch, yaw, roll) via solvePnP

It exposes a `FrameProcessor` class which accepts a BGR frame and returns a
`LandmarkFeatures` dataclass containing the computed features.

Note: MediaPipe and OpenCV are required at runtime. The exact landmark
indices are configurable constants and may be adjusted to match dataset
conventions.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:  # pragma: no cover - import-time
    mp = None


@dataclass
class LandmarkFeatures:
    """Feature container returned by :class:`FrameProcessor`.

    Attributes:
        landmarks: ndarray of shape (N, 3) with (x, y, z) normalized coordinates
        ear_left: Eye Aspect Ratio for left eye
        ear_right: Eye Aspect Ratio for right eye
        mar: Mouth Aspect Ratio
        pitch: head pitch in degrees
        yaw: head yaw in degrees
        roll: head roll in degrees
        timestamp: optional timestamp in seconds
    """
    landmarks: np.ndarray
    ear_left: float
    ear_right: float
    mar: float
    pitch: float
    yaw: float
    roll: float
    timestamp: Optional[float] = None


# Default MediaPipe FaceMesh landmark indices used for common features. These
# are widely-used approximations; adjust if you have a different mapping.
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_INDICES = [13, 14, 78, 308, 82, 312]


def compute_ear(eye: np.ndarray) -> float:
    """Compute the Eye Aspect Ratio (EAR) for a single eye.

    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

    Args:
        eye: array-like of shape (6, 2) containing the 2D coordinates for the
             eye landmarks ordered as p1..p6.

    Returns:
        EAR value (float). Returns 0.0 for degenerate inputs.
    """
    eye = np.asarray(eye, dtype=float)
    if eye.shape != (6, 2):
        raise ValueError("eye must be (6,2) array")
    # vertical distances
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    # horizontal distance
    C = np.linalg.norm(eye[0] - eye[3])
    if C == 0:
        return 0.0
    ear = (A + B) / (2.0 * C)
    return float(ear)


def compute_mar(mouth: np.ndarray) -> float:
    """Compute Mouth Aspect Ratio (MAR).

    Uses a 6-point approximation similar to EAR; ordering should match
    MOUTH_INDICES used by the caller.

    Args:
        mouth: array-like of shape (6, 2) containing mouth landmark coords.

    Returns:
        MAR value (float).
    """
    mouth = np.asarray(mouth, dtype=float)
    if mouth.shape != (6, 2):
        raise ValueError("mouth must be (6,2) array")
    A = np.linalg.norm(mouth[1] - mouth[5])
    B = np.linalg.norm(mouth[2] - mouth[4])
    C = np.linalg.norm(mouth[0] - mouth[3])
    if C == 0:
        return 0.0
    mar = (A + B) / (2.0 * C)
    return float(mar)


def estimate_head_pose(landmarks: np.ndarray, image_size: Tuple[int, int]) -> Tuple[float, float, float]:
    """Estimate head pose (pitch, yaw, roll) using solvePnP.

    This function selects a small set of 2D image points and maps them to a
    simple 3D face model. The indexes chosen are heuristics for MediaPipe
    FaceMesh and may be adjusted.

    Args:
        landmarks: array of shape (N, 3) with normalized (x, y, z) coords.
        image_size: (width, height) of the source image in pixels.

    Returns:
        Tuple (pitch, yaw, roll) in degrees.
    """
    w, h = image_size
    lm = np.asarray(landmarks)
    if lm.ndim != 2 or lm.shape[1] < 2:
        raise ValueError("landmarks must be (N, >=2) array")

    # Select points (nose tip, chin, left eye corner, right eye corner, mouth left, mouth right)
    # These indices are approximate and may be adjusted for your landmarks.
    try:
        image_points = np.array([
            lm[1][:2],   # nose tip
            lm[152][:2], # chin
            lm[33][:2],  # left eye left corner
            lm[263][:2], # right eye right corner
            lm[61][:2],  # mouth left
            lm[291][:2], # mouth right
        ])
    except Exception as e:
        raise RuntimeError("required landmark indices not found") from e

    # convert normalized coords to image pixels
    image_points[:, 0] *= w
    image_points[:, 1] *= h

    # 3D model points in an arbitrary coordinate system (approximate)
    model_points = np.array([
        (0.0, 0.0, 0.0),        # nose tip
        (0.0, -63.6, -12.5),    # chin
        (-43.3, 32.7, -26.0),   # left eye left corner
        (43.3, 32.7, -26.0),    # right eye right corner
        (-28.9, -28.9, -20.0),  # mouth left
        (28.9, -28.9, -20.0),   # mouth right
    ], dtype=float)

    focal_length = w
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=float)

    dist_coeffs = np.zeros((4, 1))
    success, rotation_vec, translation_vec = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = np.hstack((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    # euler_angles are in degrees: [pitch, yaw, roll]
    # decomposeProjectionMatrix returns euler_angles as shape (3, 1), so flatten
    euler_angles = euler_angles.flatten()
    pitch, yaw, roll = float(euler_angles[0]), float(euler_angles[1]), float(euler_angles[2])
    return pitch, yaw, roll


class FrameProcessor:
    """Process BGR frames and return `LandmarkFeatures`.

    Example:
        fp = FrameProcessor(static_image_mode=False)
        features = fp.process(frame)
    """

    def __init__(self, static_image_mode: bool = False, refine_landmarks: bool = True):
        if mp is None:
            raise RuntimeError("mediapipe is required for FrameProcessor")
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(static_image_mode=static_image_mode,
                                                      max_num_faces=1,
                                                      refine_landmarks=refine_landmarks,
                                                      min_detection_confidence=0.5,
                                                      min_tracking_confidence=0.5)

    def process(self, frame: np.ndarray, timestamp: Optional[float] = None) -> LandmarkFeatures:
        """Process a single BGR frame.

        Args:
            frame: BGR image as numpy array (H, W, 3)
            timestamp: optional timestamp

        Returns:
            LandmarkFeatures containing landmarks and computed metrics.
        """
        if frame is None:
            raise ValueError("frame is None")
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            # return empty features with zeros
            empty = np.zeros((468, 3), dtype=float)
            return LandmarkFeatures(
                landmarks=empty,
                ear_left=0.0,
                ear_right=0.0,
                mar=0.0,
                pitch=0.0,
                yaw=0.0,
                roll=0.0,
                timestamp=timestamp,
            )

        face_landmarks = results.multi_face_landmarks[0]
        lm = np.array([[p.x, p.y, getattr(p, 'z', 0.0)] for p in face_landmarks.landmark], dtype=float)

        # compute EARs
        left_eye_pts = (lm[LEFT_EYE_INDICES][:, :2] * np.array([w, h]))
        right_eye_pts = (lm[RIGHT_EYE_INDICES][:, :2] * np.array([w, h]))
        ear_l = compute_ear(left_eye_pts)
        ear_r = compute_ear(right_eye_pts)

        # compute MAR
        mouth_pts = (lm[MOUTH_INDICES][:, :2] * np.array([w, h]))
        mar = compute_mar(mouth_pts)

        # head pose
        pitch, yaw, roll = estimate_head_pose(lm, (w, h))

        return LandmarkFeatures(
            landmarks=lm,
            ear_left=ear_l,
            ear_right=ear_r,
            mar=mar,
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            timestamp=timestamp,
        )
