# VisionGuard AI v2.0

**Опис проекту:**
VisionGuard AI — інтелектуальна система відеоспостереження на Python. Використовує вебкамеру або локальні відеофайли для детекції об'єктів (людина, авто, велосипед тощо) за допомогою моделі YOLOv8. Система формує події, відображає їх у GUI з Live Video та історією, зберігає кадри подій, а також надсилає повідомлення в Telegram.

**Можливості:**
- Live Video з накладенням bounding boxes та зон.
- Налаштування зон та фільтру подій по них.
- Історія подій з мініатюрами кадрів.
- Статистика подій по класах та часу.
- Telegram сповіщення (текст + фото).
- Експорт подій у CSV та Excel.
- Антифлуд для повторних подій.

**Встановлення:**
```bash
git clone <repository_url>
cd VideoSurveillanceMVP
python -m venv venv
venv\Scripts\activate  # Windows
# або source venv/bin/activate  # Linux / Mac
pip install -r requirements.txt