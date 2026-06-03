# config.py

CAMERA_SOURCE = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

MODEL_PATH = "yolov8s.pt"

ALERT_CLASSES = ["person", "car", "bicycle", "truck", "motorcycle"]

DETECTION_CONFIDENCE_THRESHOLD = 0.25
MIN_EVENT_INTERVAL = 5

DB_PATH = "events.db"
ZONES_FILE = "zones.json"

APP_NAME = "VisionGuard AI"
APP_VERSION = "2.0"

CLASS_COLORS = {
    "person":     (0, 255, 0),
    "car":        (255, 0, 0),
    "bicycle":    (0, 255, 255),
    "truck":      (255, 128, 0),
    "motorcycle": (255, 0, 255),
    "default":    (0, 255, 0),
}

TELEGRAM_BOT_TOKEN = "8854057150:AAFgD-vSYJaE4kotJu9iCSZVoBNsXT47Eck"
CHAT_ID = "465149755"

# Якщо True — детектувати тільки об'єкти всередині зон.
# Якщо False — детектувати по всьому кадру.
ZONE_FILTER_ENABLED = True

# Папка для збереження кадрів подій
EVENTS_DIR = "events"