import os
import cv2
import numpy as np


ALLOWED_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
ALLOWED_VIDEO_EXTENSIONS = ('.mp4',)


def get_video_capture(source=0):
    """
    Відкриває тільки локальні джерела:
    - вебкамера: 0, 1, 2...
    - фото: .jpg, .jpeg, .png
    - відео: .mp4
    """

    # Вебкамера через int або рядок "0"
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        cap = cv2.VideoCapture(int(source))

        if not cap.isOpened():
            raise ValueError(f"Не вдалося відкрити вебкамеру: {source}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        return cap

    source_path = str(source)
    source_lower = source_path.lower()

    # Заборона мережевих джерел
    if source_lower.startswith(('http://', 'https://', 'rtsp://', 'rtmp://')):
        raise ValueError("Дозволені тільки локальні джерела: вебкамера або файли .jpg/.png/.mp4")

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Файл не знайдено: {source_path}")

    _, ext = os.path.splitext(source_lower)

    # Фото
    if ext in ALLOWED_IMAGE_EXTENSIONS:
        frame = cv2.imread(source_path)

        if frame is None or frame.size == 0:
            raise ValueError(f"Не вдалося прочитати зображення або кадр порожній: {source_path}")

        return frame

    # Відео
    if ext in ALLOWED_VIDEO_EXTENSIONS:
        cap = cv2.VideoCapture(source_path)

        if not cap.isOpened():
            raise ValueError(f"Не вдалося відкрити відеофайл: {source_path}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        return cap

    raise ValueError("Непідтримуваний формат. Дозволено: .jpg, .jpeg, .png, .mp4 або вебкамера")


def read_frame(cap):
    """
    Синхронне читання кадру.
    Повертає:
    - кадр, якщо він успішно прочитаний;
    - None, якщо кадр порожній або потік завершився.
    """

    if cap is None:
        return None

    # Якщо це один кадр зображення
    if isinstance(cap, np.ndarray):
        if cap.size == 0:
            return None
        return cap

    ret, frame = cap.read()

    if not ret or frame is None or frame.size == 0:
        return None

    return frame