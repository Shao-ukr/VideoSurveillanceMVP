# zones.py — Управління зонами виявлення
import json
import os
import cv2
import numpy as np

ZONES_FILE = "zones.json"


def load_zones() -> list[dict]:
    """
    Завантажує зони з JSON-файлу.
    Кожна зона: {"name": str, "points": [[x,y], ...], "enabled": bool}
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
    Використовує алгоритм ray casting.
    """
    if len(polygon) < 3:
        return False
    pts = np.array(polygon, dtype=np.int32)
    result = cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False)
    return result >= 0


def bbox_intersects_zone(bbox: list[int], zone_points: list[list[int]]) -> bool:
    """
    Перевіряє чи bounding box перетинається із зоною.
    Використовує центр bbox.
    """
    if len(bbox) != 4 or len(zone_points) < 3:
        return False

    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2  # зміна: центр bbox, а не нижній край

    return point_in_polygon((cx, cy), zone_points)


def bbox_intersects_any_zone(bbox: list[int], zones: list[dict]) -> tuple[bool, str | None]:
    """
    Перевіряє чи bbox перетинається з будь-якою увімкненою зоною.
    Повертає (True, zone_name) або (False, None).
    """
    for zone in zones:
        if not zone.get("enabled", True):
            continue
        points = zone.get("points", [])
        if bbox_intersects_zone(bbox, points):
            return True, zone.get("name", "Зона")
    return False, None


def draw_zones_on_frame(
    frame: np.ndarray,
    zones: list[dict],
    frame_w: int,
    frame_h: int,
    original_w: int = 640,
    original_h: int = 640,
    alpha: float = 0.25,
) -> np.ndarray:
    """
    Малює зони на кадрі з напівпрозорим заливанням.
    Масштабує координати зон під поточний розмір кадру.
    """
    if not zones:
        return frame

    overlay = frame.copy()
    scale_x = frame_w / original_w
    scale_y = frame_h / original_h

    zone_colors = [
        (0, 200, 255),   # жовтий
        (255, 100, 0),   # блакитний
        (0, 255, 100),   # зелений
        (200, 0, 255),   # фіолетовий
        (0, 128, 255),   # помаранчевий
    ]

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
        color = zone_colors[i % len(zone_colors)]

        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

        # Підпис зони
        if scaled:
            label_x = min(p[0] for p in scaled)
            label_y = min(p[1] for p in scaled) - 6
            label_y = max(label_y, 14)
            cv2.putText(
                frame,
                zone.get("name", f"Зона {i+1}"),
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame