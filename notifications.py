# notifications.py — Telegram сповіщення з підтримкою фото та черги
import urllib.parse
import urllib.request
import threading
import queue
import time

try:
    from config import TELEGRAM_BOT_TOKEN, CHAT_ID
except ImportError:
    try:
        from config import BOT_TOKEN as TELEGRAM_BOT_TOKEN, CHAT_ID
    except ImportError:
        TELEGRAM_BOT_TOKEN = None
        CHAT_ID = None

# ── Асинхронна черга сповіщень ──────────────────────────────────────────────

_notification_queue: queue.Queue = queue.Queue()
_worker_thread: threading.Thread | None = None
_last_group_time: float = 0.0
_pending_group: list[str] = []
_GROUP_INTERVAL = 10.0  # секунди: групуємо сповіщення


def _worker():
    """Фоновий потік обробки черги сповіщень."""
    global _last_group_time, _pending_group
    while True:
        try:
            item = _notification_queue.get(timeout=_GROUP_INTERVAL)
            if item is None:
                break

            kind, payload = item

            if kind == "text":
                _pending_group.append(payload)
            elif kind == "photo":
                # Фото відправляємо одразу
                path, caption = payload
                _send_photo_sync(path, caption)

            # Групуємо текстові повідомлення
            now = time.time()
            if _pending_group and (now - _last_group_time >= _GROUP_INTERVAL):
                grouped = "\n\n".join(_pending_group)
                _send_message_sync(grouped)
                _pending_group.clear()
                _last_group_time = now

            _notification_queue.task_done()

        except queue.Empty:
            # Таймаут — відправляємо накопичені повідомлення
            if _pending_group:
                grouped = "\n\n".join(_pending_group)
                _send_message_sync(grouped)
                _pending_group.clear()
                _last_group_time = time.time()


def _ensure_worker():
    global _worker_thread
    if _worker_thread is None or not _worker_thread.is_alive():
        _worker_thread = threading.Thread(target=_worker, daemon=True)
        _worker_thread.start()


# ── Sync відправка (для worker) ─────────────────────────────────────────────

def _send_message_sync(message: str) -> bool:
    try:
        if not TELEGRAM_BOT_TOKEN or not CHAT_ID:
            return False
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": CHAT_ID,
            "text": message,
            "parse_mode": "HTML",
        }).encode("utf-8")
        req = urllib.request.Request(url=url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=8) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"Telegram помилка: {e}")
        return False


def _send_photo_sync(photo_path: str, caption: str = "") -> bool:
    """Надсилає фото через multipart/form-data (RFC 2046)."""
    try:
        if not TELEGRAM_BOT_TOKEN or not CHAT_ID:
            return False

        import os
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        boundary = b"----VisionGuardBoundary7MA4YWxkTrZu0gW"

        def field(name: str, value: str) -> bytes:
            return (
                b"--" + boundary + b"\r\n"
                b'Content-Disposition: form-data; name="' + name.encode() + b'"\r\n\r\n'
                + value.encode("utf-8") + b"\r\n"
            )

        with open(photo_path, "rb") as f:
            photo_data = f.read()

        fname = os.path.basename(photo_path).encode()
        photo_part = (
            b"--" + boundary + b"\r\n"
            b'Content-Disposition: form-data; name="photo"; filename="' + fname + b'"\r\n'
            b"Content-Type: image/jpeg\r\n\r\n"
            + photo_data + b"\r\n"
        )

        body = (
            field("chat_id", str(CHAT_ID))
            + (field("caption", caption[:1024]) if caption else b"")
            + photo_part
            + b"--" + boundary + b"--\r\n"
        )

        req = urllib.request.Request(
            url=url,
            data=body,
            method="POST",
            headers={"Content-Type": "multipart/form-data; boundary=" + boundary.decode()},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"Telegram фото помилка: {e}")
        return False


# ── Публічний API ────────────────────────────────────────────────────────────

def send_telegram_message(message: str) -> bool:
    """
    Додає текстове повідомлення до черги (асинхронно, з групуванням).
    """
    _ensure_worker()
    _notification_queue.put(("text", message))
    return True


def send_telegram_photo(photo_path: str, caption: str = "") -> bool:
    """
    Надсилає фото події через Telegram (асинхронно).
    """
    _ensure_worker()
    _notification_queue.put(("photo", (photo_path, caption)))
    return True


def send_telegram_message_sync(message: str) -> bool:
    """Синхронне надсилання (для тестування)."""
    return _send_message_sync(message)