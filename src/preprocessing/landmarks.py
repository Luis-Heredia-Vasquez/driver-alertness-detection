import cv2
import numpy as np


def extract_face_landmarks(frame):
    """Placeholder: return dummy landmarks for a frame"""
    h, w = frame.shape[:2]
    # return normalized center point as example
    return np.array([[w / 2 / w, h / 2 / h]])


def preprocess_for_model(frame, target_size=(64, 64)):
    img = cv2.resize(frame, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.transpose(img, (2, 0, 1))
    return img
