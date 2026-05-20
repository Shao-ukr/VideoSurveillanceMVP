# test_all.py
import sys

print("=== Перевірка numpy ===")
try:
    import numpy as np
    a = np.array([1, 2, 3])
    print("numpy працює:", a)
except Exception as e:
    print("Помилка numpy:", e)

print("\n=== Перевірка OpenCV ===")
try:
    import cv2
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        print("OpenCV працює, розмір кадру:", frame.shape)
    else:
        print("OpenCV працює, але камера не доступна")
except Exception as e:
    print("Помилка OpenCV:", e)

print("\n=== Перевірка PyQt5 ===")
try:
    from PyQt5.QtWidgets import QApplication, QLabel
    app = QApplication(sys.argv)
    label = QLabel("PyQt5 працює")
    # Не показуємо вікно, лише перевірка імпорту
    print("PyQt5 імпорт успішний")
except Exception as e:
    print("Помилка PyQt5:", e)

print("\n=== Перевірка PyTorch ===")
try:
    import torch
    import torchvision
    x = torch.rand(2, 3)
    print("PyTorch працює на пристрої:", x.device)
except Exception as e:
    print("Помилка PyTorch:", e)

print("\n=== Перевірка python-telegram-bot ===")
try:
    from telegram import Bot
    print("Telegram Bot імпорт успішний")
except Exception as e:
    print("Помилка Telegram Bot:", e)

print("\n=== ВСЕ ПЕРЕВІРЕНО ===")