# events.py
import os
import cv2
import time
import datetime

from config import ALERT_CLASSES, DETECTION_CONFIDENCE_THRESHOLD, ZONE_FILTER_ENABLED
from notifications import send_telegram_message, send_telegram_photo
from db import log_event, can_send_event
from zones import bbox_intersects_any_zone

EVENTS_DIR = "events"

EVENT_TYPE_MAP = {
    "person":     "Вторгнення людини",
    "car":        "Автомобіль у зоні",
    "bicycle":    "Велосипед у зоні",
    "truck":      "Вантажівка у зоні",
    "motorcycle": "Мотоцикл у зоні",
}

# Типи подій на основі зони
ZONE_EVENT_TYPE_MAP = {
    "critical":  "🔴 КРИТИЧНА ТРИВОГА",
    "secondary": "🟡 Виявлення",
    "test":      "🟢 Тест",
}


def get_event_type(object_name: str, zone_type: str = "secondary") -> str:
    """Перетворює клас YOLO та тип зони у зрозумілий тип події."""
    base = EVENT_TYPE_MAP.get(object_name, f"Виявлено: {object_name}")
    prefix = ZONE_EVENT_TYPE_MAP.get(zone_type, "")
    return f"{prefix} {base}" if prefix else base


def save_event_frame(frame, bbox: list[int], object_name: str, frame_number: int) -> str | None:
    """
    Зберігає кадр події у папку events/.
    Повертає шлях до збереженого файлу або None.
    """
    try:
        os.makedirs(EVENTS_DIR, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_{object_name}_f{frame_number}.jpg"
        path = os.path.join(EVENTS_DIR, filename)

        frame_copy = frame.copy() if frame is not None else None
        if frame_copy is None:
            return None

        # Малюємо bbox на збереженому кадрі
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{object_name}"
            cv2.putText(frame_copy, label, (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imwrite(path, frame_copy)
        return path
    except Exception as e:
        print(f"Помилка збереження кадру події: {e}")
        return None


def generate_events(
    objects: list,
    frame_number: int,
    camera_id: str = "CAM-01",
    confidence_threshold: float | None = None,
    event_interval: float | None = None,
    send_notifications: bool = True,
    zones: list | None = None,
    raw_frame=None,   # оригінальний кадр для збереження
) -> list[dict]:
    """
    Генерує події для виявлених об'єктів.

    Якщо zones передані та ZONE_FILTER_ENABLED=True —
    враховуються тільки об'єкти всередині зон.

    Повертає список подій + активні зони (для підсвічування).
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
        zone_type = "secondary"
        if use_zones:
            in_zone, zone_name = bbox_intersects_any_zone(bbox, active_zones)
            if not in_zone:
                continue  # об'єкт поза зоною — пропустити

            # Знаходимо тип зони
            for z in active_zones:
                if z.get("name") == zone_name:
                    zone_type = z.get("zone_type", "secondary")
                    break
        else:
            zone_name = None

        cooldown_key = f"{camera_id}:{object_name}"

        if not can_send_event(cooldown_key, interval=event_interval):
            continue

        event_type = get_event_type(object_name, zone_type)
        if zone_name:
            event_type = f"{event_type} [{zone_name}]"

        # Зберігаємо кадр події
        frame_path = None
        if raw_frame is not None:
            frame_path = save_event_frame(raw_frame, bbox, object_name, frame_number)

        event = {
            "frame": frame_number,
            "object_class": object_name,
            "bbox": bbox,
            "confidence": confidence,
            "event_type": event_type,
            "camera_id": camera_id,
            "zone_name": zone_name,
            "zone_type": zone_type,
            "frame_path": frame_path,
            "timestamp": time.time(),
        }

        events.append(event)

        log_event(
            frame=frame_number,
            object_class=object_name,
            event_type=event_type,
            bbox=bbox,
            confidence=confidence,
            camera_id=camera_id,
            frame_path=frame_path,
        )

        if send_notifications:
            zone_info = f"\nЗона: {zone_name}" if zone_name else ""
            message = (
                f"{event_type}\n"
                f"Камера: {camera_id} | Кадр: {frame_number}{zone_info}\n"
                f"Впевненість: {confidence:.0%}"
            )
            if frame_path and os.path.exists(frame_path):
                send_telegram_photo(frame_path, caption=message)
            else:
                send_telegram_message(message)

    return events