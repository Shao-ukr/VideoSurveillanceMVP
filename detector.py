# detector.py
from ultralytics import YOLO
from config import MODEL_PATH, ALERT_CLASSES, DETECTION_CONFIDENCE_THRESHOLD

model = YOLO(MODEL_PATH)

try:
    model.to("cpu")
except Exception:
    pass


def detect_objects(frame, confidence_threshold=None):
    """
    Детекція об'єктів на кадрі.
    """
    if frame is None:
        return []

    threshold = (
        confidence_threshold
        if confidence_threshold is not None
        else DETECTION_CONFIDENCE_THRESHOLD
    )

    results = model.predict(
        frame,
        verbose=False,
        device="cpu"
    )[0]

    detected_objects = []

    if results.boxes is None:
        return detected_objects

    for box, cls, conf in zip(
        results.boxes.xyxy,
        results.boxes.cls,
        results.boxes.conf
    ):
        class_id = int(cls.item()) if hasattr(cls, "item") else int(cls)
        confidence = float(conf.item()) if hasattr(conf, "item") else float(conf)

        class_name = model.names[class_id]

        if class_name not in ALERT_CLASSES:
            continue

        if confidence < threshold:
            continue

        bbox = [int(x) for x in box.tolist()]

        detected_objects.append({
            "class": class_id,
            "name": class_name,
            "confidence": confidence,
            "bbox": bbox,
        })

    return detected_objects