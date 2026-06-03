# preprocessing.py
import cv2


def preprocess_frame(frame, size=(640, 640)):
    if frame is None or frame.size == 0:
        return None
    frame_resized = cv2.resize(frame, size)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_filtered = cv2.GaussianBlur(frame_rgb, (3, 3), 0)
    return frame_filtered