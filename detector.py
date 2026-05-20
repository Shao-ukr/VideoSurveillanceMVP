from ultralytics import YOLO
from config import MODEL_PATH, ALERT_CLASSES


try:
    from config import DETECTION_CONFIDENCE_THRESHOLD
except ImportError:
    DETECTION_CONFIDENCE_THRESHOLD = 0.25


model = YOLO(MODEL_PATH)

# Примусово переводимо модель на CPU.
# У більшості версій ultralytics безпечніше також передавати device="cpu" у predict().
try:
    model.to("cpu")
except Exception:
    pass


def detect_objects(frame):
    """
    Детекція об'єктів на кадрі.

    Повертає список:
    [
        {
            'class': class_id,
            'name': class_name,
            'confidence': score,
            'bbox': [x1, y1, x2, y2]
        }
    ]
    """

    if frame is None:
        return []

    results = model.predict(
        frame,
        verbose=False,
        device="cpu"
    )[0]

    detected_objects = []

    if results.boxes is None:
        return detected_objects

    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        class_id = int(cls.item()) if hasattr(cls, "item") else int(cls)
        confidence = float(conf.item()) if hasattr(conf, "item") else float(conf)

        class_name = model.names[class_id]

        if class_name not in ALERT_CLASSES:
            continue

        if confidence < DETECTION_CONFIDENCE_THRESHOLD:
            continue

        bbox = [int(x) for x in box.tolist()]

        detected_objects.append({
            'class': class_id,
            'name': class_name,
            'confidence': confidence,
            'bbox': bbox
        })

    return detected_objects