# events.py
from config import ALERT_CLASSES, DETECTION_CONFIDENCE_THRESHOLD, ZONE_FILTER_ENABLED
from notifications import send_telegram_message
from db import log_event, can_send_event
from zones import bbox_intersects_any_zone


EVENT_TYPE_MAP = {
    "person": "Людина у кадрі",
    "car": "Автомобіль у кадрі",
    "bicycle": "Велосипед у кадрі",
    "truck": "Вантажівка у кадрі",
    "motorcycle": "Мотоцикл у кадрі",
}


def get_event_type(object_name: str) -> str:
    """Перетворює клас YOLO у зрозумілий тип події."""
    return EVENT_TYPE_MAP.get(
        object_name,
        f"Виявлено об'єкт: {object_name}"
    )


def generate_events(
    objects: list,
    frame_number: int,
    camera_id: str = "CAM-01",
    confidence_threshold: float | None = None,
    event_interval: float | None = None,
    send_notifications: bool = True,
    zones: list | None = None,
) -> list[dict]:
    """
    Генерує події для виявлених об'єктів.

    Якщо zones передані та ZONE_FILTER_ENABLED=True —
    враховуються тільки об'єкти всередині зон.
    """
    threshold = (
        confidence_threshold
        if confidence_threshold is not None
        else DETECTION_CONFIDENCE_THRESHOLD
    )

    active_zones = zones if zones else []
    use_zones = ZONE_FILTER_ENABLED and len(active_zones) > 0

    events = []

    for obj in objects:
        object_name = obj.get("name")
        confidence = float(obj.get("confidence", 0))
        bbox = obj.get("bbox", [])

        if object_name not in ALERT_CLASSES:
            continue

        if confidence < threshold:
            continue

        # Перевірка зон
        zone_name = None
        if use_zones:
            in_zone, zone_name = bbox_intersects_any_zone(bbox, active_zones)
            if not in_zone:
                continue  # об'єкт поза зоною — пропустити

        cooldown_key = f"{camera_id}:{object_name}"

        if not can_send_event(cooldown_key, interval=event_interval):
            continue

        event_type = get_event_type(object_name)

        # Додаємо назву зони до типу події якщо є
        if zone_name:
            event_type = f"{event_type} [{zone_name}]"

        event = {
            "frame": frame_number,
            "object_class": object_name,
            "bbox": bbox,
            "confidence": confidence,
            "event_type": event_type,
            "camera_id": camera_id,
            "zone_name": zone_name,
        }

        events.append(event)

        log_event(
            frame=frame_number,
            object_class=object_name,
            event_type=event_type,
            bbox=bbox,
            confidence=confidence,
            camera_id=camera_id,
        )

        if send_notifications:
            zone_info = f"\nЗона: {zone_name}" if zone_name else ""
            message = (
                f"🔔 {event_type}\n"
                f"Камера: {camera_id} | Кадр: {frame_number}{zone_info}\n"
                f"Впевненість: {confidence:.0%}"
            )
            send_telegram_message(message)

    return events