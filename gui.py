import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QTextEdit, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

from camera import get_video_capture, read_frame
from preprocessing import preprocess_frame
from detector import detect_objects
from events import generate_events
from db import init_db


def draw_boxes(frame, objects, model_size=(640, 640)):
    """
    Накладає bounding boxes на кадр.
    Масштабує bbox з моделі 640x640 до розміру оригінального кадру.
    """
    if frame is None or frame.size == 0:
        return frame

    output_frame = frame.copy()
    frame_h, frame_w = output_frame.shape[:2]
    scale_x = frame_w / model_size[0]
    scale_y = frame_h / model_size[1]

    for obj in objects:
        bbox = obj.get('bbox', [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(coord * scale) for coord, scale in zip(bbox, [scale_x, scale_y, scale_x, scale_y])]
        name = obj.get('name', 'object')
        confidence = float(obj.get('confidence', 0))
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output_frame, f"{name}:{confidence:.2f}", (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return output_frame


class VideoSurveillanceGUI(QWidget):
    def __init__(self):
        super().__init__()
        init_db()  # Створюємо БД на старті

        self.setWindowTitle("Intelligent Video Surveillance MVP")
        self.setGeometry(100, 100, 1200, 700)

        self.cap = None
        self.frame_number = 0
        self.single_image_mode = False
        self.model_frame_size = (640, 640)

        main_layout = QHBoxLayout()
        video_layout = QVBoxLayout()
        controls_layout = QVBoxLayout()

        # Відео
        self.label = QLabel("Video Stream")
        self.label.setMinimumSize(640, 480)
        video_layout.addWidget(self.label)

        # Термінал
        self.terminal = QTextEdit()
        self.terminal.setReadOnly(True)
        video_layout.addWidget(self.terminal)

        # Кнопка камери
        self.btn_camera = QPushButton("Запустити камеру")
        self.btn_camera.clicked.connect(self.start_camera)
        controls_layout.addWidget(self.btn_camera)

        # Кнопка файлу
        self.btn_file = QPushButton("Завантажити відео/кадр")
        self.btn_file.clicked.connect(self.load_file)
        controls_layout.addWidget(self.btn_file)

        # Заглушка зон
        self.btn_zones = QPushButton("Налаштувати зони")
        self.btn_zones.clicked.connect(lambda: self.terminal.append("Налаштування зон поки недоступне"))
        controls_layout.addWidget(self.btn_zones)

        main_layout.addLayout(video_layout, 3)
        main_layout.addLayout(controls_layout, 1)
        self.setLayout(main_layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def release_current_source(self):
        self.timer.stop()
        if self.cap is not None and hasattr(self.cap, "release"):
            self.cap.release()
        self.cap = None
        self.single_image_mode = False

    def start_camera(self):
        try:
            self.release_current_source()
            self.cap = get_video_capture(0)
            self.frame_number = 0
            self.single_image_mode = False
            self.terminal.append("Камеру запущено")
            self.timer.start(30)
        except Exception as e:
            self.terminal.append(f"Помилка підключення камери: {e}")

    def load_file(self):
        try:
            path, _ = QFileDialog.getOpenFileName(self, "Open Video File or Frame", "", "Media Files (*.jpg *.jpeg *.png *.mp4)")
            if not path:
                return
            self.release_current_source()
            self.cap = get_video_capture(path)
            self.single_image_mode = path.lower().endswith(('.jpg', '.jpeg', '.png'))
            self.frame_number = 0
            self.terminal.append(f"Файл завантажено: {path}")
            self.timer.start(30)
        except Exception as e:
            self.terminal.append(f"Помилка завантаження файлу: {e}")

    def update_frame(self):
        if self.cap is None:
            return
        frame = read_frame(self.cap)
        if frame is None:
            self.timer.stop()
            self.terminal.append("Кадр недоступний або потік завершено")
            return
        frame_proc = preprocess_frame(frame, size=self.model_frame_size)
        if frame_proc is None:
            self.timer.stop()
            self.terminal.append("Кадр не оброблено")
            return

        objects = detect_objects(frame_proc)
        events = generate_events(objects, self.frame_number)

        self.terminal.append(f"--- Кадр {self.frame_number} ---")
        if events:
            for event in events:
                self.terminal.append(
                    f"Подія: {event['event_type']} | "
                    f"Об'єкт: {event['object_class']} | "
                    f"bbox: {event['bbox']} | "
                    f"ймовірність: {event['confidence']:.2f}"
                )
        else:
            self.terminal.append("Подій не виявлено")

        self.frame_number += 1

        # Малюємо bounding boxes
        frame_with_boxes = draw_boxes(frame, objects, model_size=self.model_frame_size)
        frame_rgb = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(q_img))

        if self.single_image_mode:
            self.timer.stop()

    def closeEvent(self, event):
        self.release_current_source()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = VideoSurveillanceGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()