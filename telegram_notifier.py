import asyncio
from telegram import Bot

# Встав сюди свій токен бота
BOT_TOKEN = "8854057150:AAFgD-vSYJaE4kotJu9iCSZVoBNsXT47Eck"

bot = Bot(token=BOT_TOKEN)

async def get_chat_ids():
    updates = await bot.get_updates()
    if not updates:
        print("Немає нових оновлень. Надішли повідомлення боту в Telegram.")
    for u in updates:
        if u.message:
            print(f"chat_id: {u.message.chat.id}  |  username: {u.message.chat.username}")

# Запуск асинхронної функції
asyncio.run(get_chat_ids())