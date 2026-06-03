# db.py
import sqlite3
import time
import json
from config import DB_PATH, MIN_EVENT_INTERVAL

last_event_time = {}


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Створює БД та таблицю подій.
    Якщо таблиця вже існує, перевіряє наявність потрібних колонок.
    """
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            frame INTEGER,
            object_class TEXT,
            event_type TEXT,
            bbox TEXT,
            confidence REAL,
            timestamp REAL
        )
    """)

    c.execute("PRAGMA table_info(events)")
    existing_columns = [row["name"] for row in c.fetchall()]

    if "camera_id" not in existing_columns:
        c.execute("ALTER TABLE events ADD COLUMN camera_id TEXT DEFAULT 'CAM-01'")

    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_timestamp
        ON events(timestamp)
    """)

    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_object_class
        ON events(object_class)
    """)

    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_camera_id
        ON events(camera_id)
    """)

    conn.commit()
    conn.close()


def log_event(frame, object_class, event_type, bbox, confidence, camera_id="CAM-01"):
    """
    Логує подію в SQLite.
    bbox зберігається у JSON-форматі.
    """
    conn = get_connection()
    c = conn.cursor()

    timestamp = time.time()
    bbox_json = json.dumps(bbox, ensure_ascii=False)

    c.execute("""
        INSERT INTO events (
            frame,
            object_class,
            event_type,
            bbox,
            confidence,
            timestamp,
            camera_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        frame,
        object_class,
        event_type,
        bbox_json,
        confidence,
        timestamp,
        camera_id,
    ))

    conn.commit()
    conn.close()


def can_send_event(object_class, interval=None):
    """
    Антифлуд-фільтр.
    Не дозволяє занадто часто логувати й надсилати одну й ту саму подію.
    """
    threshold = interval if interval is not None else MIN_EVENT_INTERVAL
    now = time.time()

    last_time = last_event_time.get(object_class, 0)

    if now - last_time >= threshold:
        last_event_time[object_class] = now
        return True

    return False


def get_events(
    limit=200,
    object_class_filter=None,
    event_type_filter=None,
    min_confidence=0.0,
    date_from=None,
    date_to=None,
    camera_id_filter=None,
):
    """
    Повертає список подій з фільтрацією.
    """
    conn = get_connection()
    c = conn.cursor()

    query = "SELECT * FROM events WHERE 1=1"
    params = []

    if object_class_filter:
        query += " AND object_class = ?"
        params.append(object_class_filter)

    if event_type_filter:
        query += " AND event_type LIKE ?"
        params.append(f"%{event_type_filter}%")

    if min_confidence > 0.0:
        query += " AND confidence >= ?"
        params.append(min_confidence)

    if date_from is not None:
        query += " AND timestamp >= ?"
        params.append(date_from)

    if date_to is not None:
        query += " AND timestamp <= ?"
        params.append(date_to)

    if camera_id_filter:
        query += " AND camera_id = ?"
        params.append(camera_id_filter)

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    c.execute(query, params)
    rows = [dict(row) for row in c.fetchall()]

    conn.close()
    return rows


def get_stats():
    """
    Повертає статистику подій.
    """
    conn = get_connection()
    c = conn.cursor()

    now = time.time()

    c.execute("SELECT COUNT(*) FROM events")
    total = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM events WHERE timestamp >= ?", (now - 3600,))
    last_hour = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM events WHERE timestamp >= ?", (now - 86400,))
    last_day = c.fetchone()[0]

    c.execute("""
        SELECT object_class, COUNT(*) AS cnt
        FROM events
        GROUP BY object_class
        ORDER BY cnt DESC
    """)
    by_class = {row[0]: row[1] for row in c.fetchall()}

    conn.close()

    return {
        "total": total,
        "last_hour": last_hour,
        "last_day": last_day,
        "by_class": by_class,
    }


def delete_event(event_id):
    """
    Видаляє одну подію за ID.
    """
    conn = get_connection()
    c = conn.cursor()

    c.execute("DELETE FROM events WHERE id = ?", (event_id,))

    conn.commit()
    conn.close()


def clear_all_events():
    """
    Очищає всі події з БД.
    """
    conn = get_connection()
    c = conn.cursor()

    c.execute("DELETE FROM events")

    conn.commit()
    conn.close()