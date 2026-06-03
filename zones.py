# zones.py — Управління зонами виявлення
import json
import os
import cv2
import numpy as np

ZONES_FILE = "zones.json"

# Типи зон та їх кольори і пріоритети
ZONE_TYPES = {
    "critical":   {"label": "Критична",   "color": (0, 0, 255),   "priority": 3},
    "secondary":  {"label": "Вторинна",   "color": (0, 165, 255), "priority": 2},
    "test":       {"label": "Тестова",    "color": (0, 255, 100), "priority": 1},
}


def load_zones() -> list[dict]:
    """
    Завантажує зони з JSON-файлу.
    Кожна зона: {"name": str, "points": [[x,y], ...], "enabled": bool, "zone_type": str}
    """
    if not os.path.exists(ZONES_FILE):
        return []
    try:
        with open(ZONES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"Помилка завантаження зон: {e}")
        return []


def save_zones(zones: list[dict]) -> bool:
    """Зберігає зони у JSON-файл."""
    try:
        with open(ZONES_FILE, "w", encoding="utf-8") as f:
            json.dump(zones, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Помилка збереження зон: {e}")
        return False


def point_in_polygon(point: tuple[int, int], polygon: list[list[int]]) -> bool:
    """
    Перевіряє чи точка знаходиться всередині полігону.
    Використовує OpenCV pointPolygonTest.
    """
    if len(polygon) < 3:
        return False
    pts = np.array(polygon, dtype=np.int32)
    result = cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False)
    return result >= 0


def bbox_intersects_zone(bbox: list[int], zone_points: list[list[int]]) -> bool:
    """
    Перевіряє чи bounding box перетинається із зоною.
    ВИПРАВЛЕНО: перевіряє центр нижньої половини bbox (більш природно для людей/машин),
    а також кути bbox для кращого покриття.
    Координати bbox та zone_points мають бути в одному просторі (наприклад, 640x640).
    """
    if len(bbox) != 4 or len(zone_points) < 3:
        return False

    x1, y1, x2, y2 = bbox

    # Перевіряємо кілька контрольних точок bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    cy_bottom = y1 + int((y2 - y1) * 0.75)  # нижня чверть

    check_points = [
        (cx, cy),          # центр
        (cx, cy_bottom),   # низ центру
        (x1, y2),          # нижній лівий кут
        (x2, y2),          # нижній правий кут
    ]

    for pt in check_points:
        if point_in_polygon(pt, zone_points):
            return True
    return False


def bbox_intersects_any_zone(bbox: list[int], zones: list[dict]) -> tuple[bool, str | None]:
    """
    Перевіряє чи bbox перетинається з будь-якою увімкненою зоною.
    Повертає (True, zone_name) або (False, None).
    Враховує пріоритет зони (critical > secondary > test).
    """
    matched = []
    for zone in zones:
        if not zone.get("enabled", True):
            continue
        points = zone.get("points", [])
        if bbox_intersects_zone(bbox, points):
            zone_type = zone.get("zone_type", "secondary")
            priority = ZONE_TYPES.get(zone_type, {}).get("priority", 1)
            matched.append((priority, zone.get("name", "Зона"), zone_type))

    if not matched:
        return False, None

    # Повертаємо зону з найвищим пріоритетом
    matched.sort(reverse=True, key=lambda x: x[0])
    _, zone_name, zone_type = matched[0]
    return True, zone_name


def get_zone_color(zone: dict) -> tuple[int, int, int]:
    """Повертає колір зони залежно від її типу."""
    zone_type = zone.get("zone_type", "secondary")
    return ZONE_TYPES.get(zone_type, ZONE_TYPES["secondary"])["color"]


def draw_zones_on_frame(
    frame: np.ndarray,
    zones: list[dict],
    frame_w: int,
    frame_h: int,
    original_w: int = 640,
    original_h: int = 640,
    alpha: float = 0.22,
    active_zone_names: set = None,
) -> np.ndarray:
    """
    Малює зони на кадрі з напівпрозорим заливанням.
    Масштабує координати зон під поточний розмір кадру.
    Активні зони (де є об'єкти) підсвічуються яскравіше.
    """
    if not zones:
        return frame

    overlay = frame.copy()
    scale_x = frame_w / original_w
    scale_y = frame_h / original_h

    if active_zone_names is None:
        active_zone_names = set()

    for i, zone in enumerate(zones):
        if not zone.get("enabled", True):
            continue
        points = zone.get("points", [])
        if len(points) < 3:
            continue

        scaled = [
            [int(p[0] * scale_x), int(p[1] * scale_y)]
            for p in points
        ]
        pts = np.array(scaled, dtype=np.int32)
        color = get_zone_color(zone)

        is_active = zone.get("name", "") in active_zone_names
        fill_alpha = 0.40 if is_active else alpha

        cv2.fillPoly(overlay, [pts], color)

        border_thickness = 3 if is_active else 2
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=border_thickness)

        # Підпис зони з типом
        if scaled:
            label_x = min(p[0] for p in scaled)
            label_y = min(p[1] for p in scaled) - 6
            label_y = max(label_y, 14)

            zone_type = zone.get("zone_type", "secondary")
            type_label = ZONE_TYPES.get(zone_type, {}).get("label", "")
            active_marker = " 🔴" if is_active else ""
            display_name = f"{zone.get('name', f'Зона {i+1}')} [{type_label}]{active_marker}"

            # Фон для тексту
            (tw, th), _ = cv2.getTextSize(display_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (label_x - 2, label_y - th - 4), (label_x + tw + 4, label_y + 2),
                          (0, 0, 0), -1)
            cv2.putText(
                frame,
                display_name,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

    cv2.addWeighted(overlay, fill_alpha, frame, 1 - fill_alpha, 0, frame)
    return frame