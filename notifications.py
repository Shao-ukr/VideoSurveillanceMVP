import urllib.parse
import urllib.request


try:
    from config import TELEGRAM_BOT_TOKEN, CHAT_ID
except ImportError:
    try:
        from config import BOT_TOKEN as TELEGRAM_BOT_TOKEN, CHAT_ID
    except ImportError:
        TELEGRAM_BOT_TOKEN = None
        CHAT_ID = None


def send_telegram_message(message):
    """
    Синхронне надсилання Telegram-повідомлення через HTTP API.

    Це прибирає проблему:
    RuntimeWarning: coroutine 'Bot.send_message' was never awaited
    """

    try:
        if not TELEGRAM_BOT_TOKEN or not CHAT_ID:
            print("Telegram не налаштований: перевір TELEGRAM_BOT_TOKEN і CHAT_ID у config.py")
            return False

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

        data = urllib.parse.urlencode({
            "chat_id": CHAT_ID,
            "text": message
        }).encode("utf-8")

        request = urllib.request.Request(
            url=url,
            data=data,
            method="POST"
        )

        with urllib.request.urlopen(request, timeout=5) as response:
            if response.status != 200:
                print(f"Помилка Telegram: HTTP {response.status}")
                return False

        return True

    except Exception as e:
        print("Помилка Telegram:", e)
        return False