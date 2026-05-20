from config import ALERT_CLASSES, DETECTION_CONFIDENCE_THRESHOLD
from notifications import send_telegram_message
from db import log_event, can_send_event


def get_event_type(object_name):
    """
    Перетворює клас YOLO у зрозумілий тип події.
    """
    if object_name == "person":
        return "Людина у кадрі"
    if object_name == "car":
        return "Машина у кадрі"
    return f"Виявлено об'єкт: {object_name}"


def generate_events(objects, frame_number):
    """
    Генерує події для виявлених об'єктів.

    - Фільтрує по ALERT_CLASSES і confidence
    - Перевіряє MIN_EVENT_INTERVAL через can_send_event
    - Логує у SQLite
    - Відправляє синхронні Telegram-повідомлення
    """
    events = []

    for obj in objects:
        object_name = obj.get('name')
        confidence = float(obj.get('confidence', 0))

        # Фільтр по класу та ймовірності
        if object_name not in ALERT_CLASSES:
            continue
        if confidence < DETECTION_CONFIDENCE_THRESHOLD:
            continue

        event_type = get_event_type(object_name)

        # Фільтр флуду
        if not can_send_event(object_name):
            continue

        event = {
            'frame': frame_number,
            'object_class': object_name,
            'bbox': obj.get('bbox', []),
            'confidence': confidence,
            'event_type': event_type
        }
        events.append(event)

        # Логування в БД
        log_event(frame_number, object_name, event_type, obj.get('bbox', []), confidence)

        # Відправка повідомлення в Telegram
        message = (
            f"Подія: {event_type} | "
            f"Кадр: {frame_number} | "
            f"Об'єкт: {object_name} | "
            f"bbox: {event['bbox']} | "
            f"Ймовірність: {confidence:.2f}"
        )
        send_telegram_message(message)

    return events