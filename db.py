import sqlite3
import time

DB_PATH = "events.db"

def init_db():
    """
    Створює БД та таблицю подій, якщо ще не існує.
    """
    conn = sqlite3.connect(DB_PATH)
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
    conn.commit()
    conn.close()


def log_event(frame, object_class, event_type, bbox, confidence):
    """
    Логування події у БД SQLite.
    bbox зберігається як текст.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    timestamp = time.time()
    c.execute("""
        INSERT INTO events (frame, object_class, event_type, bbox, confidence, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (frame, object_class, event_type, str(bbox), confidence, timestamp))
    conn.commit()
    conn.close()


# Фільтр повторних подій
MIN_EVENT_INTERVAL = 5  # секунд
last_event_time = {}

def can_send_event(object_class):
    """
    Перевіряє, чи можна надсилати нове повідомлення для класу об'єкта
    на основі MIN_EVENT_INTERVAL.
    """
    now = time.time()
    last_time = last_event_time.get(object_class, 0)
    if now - last_time >= MIN_EVENT_INTERVAL:
        last_event_time[object_class] = now
        return True
    return False