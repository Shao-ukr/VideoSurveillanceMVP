# gui.py — VisionGuard AI — оновлений GUI з підтримкою зон
import sys
import cv2
import datetime

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFileDialog,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QTabWidget,
    QLineEdit,
    QGroupBox,
    QFormLayout,
    QSlider,
    QCheckBox,
    QSizePolicy,
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer

from camera import get_video_capture, read_frame
from preprocessing import preprocess_frame
from detector import detect_objects
from events import generate_events
from db import init_db, get_events, get_stats, delete_event, clear_all_events
from notifications import send_telegram_message
from zones import load_zones, draw_zones_on_frame

try:
    from config import APP_NAME, APP_VERSION
except ImportError:
    APP_NAME = "VisionGuard AI"
    APP_VERSION = "1.1"

try:
    from config import CLASS_COLORS
except ImportError:
    CLASS_COLORS = {"person": (0, 255, 0), "car": (255, 0, 0), "default": (0, 255, 255)}

try:
    from config import DETECTION_CONFIDENCE_THRESHOLD
except ImportError:
    DETECTION_CONFIDENCE_THRESHOLD = 0.25

try:
    from config import MIN_EVENT_INTERVAL
except ImportError:
    MIN_EVENT_INTERVAL = 5

try:
    from config import ZONE_FILTER_ENABLED
except ImportError:
    ZONE_FILTER_ENABLED = True


# ─────────────────────────────────────────────────────────────
# VideoPanel
# ─────────────────────────────────────────────────────────────

class VideoPanel(QLabel):
    """Панель відображення відео з накладенням bounding boxes і зон."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.setText("Відеопотік не запущено")
        self.setStyleSheet("""
            QLabel {
                background-color: #111;
                color: #666;
                border: 1px solid #333;
                font-size: 16px;
            }
        """)

    def set_frame(
        self,
        frame_bgr,
        objects=None,
        zones=None,
        model_size=(640, 640),
        show_zones: bool = True,
    ):
        if frame_bgr is None or frame_bgr.size == 0:
            return

        frame_draw = frame_bgr.copy()
        h, w = frame_draw.shape[:2]

        # Малюємо зони
        if show_zones and zones:
            frame_draw = draw_zones_on_frame(
                frame_draw, zones, w, h,
                original_w=model_size[0],
                original_h=model_size[1],
            )

        # Малюємо bounding boxes
        if objects:
            self._draw_boxes(frame_draw, objects, model_size)

        frame_rgb = cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        q_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(pixmap)

    def _draw_boxes(self, frame, objects, model_size=(640, 640)):
        frame_h, frame_w = frame.shape[:2]
        model_w, model_h = model_size
        scale_x = frame_w / model_w
        scale_y = frame_h / model_h

        for obj in objects:
            bbox = obj.get("bbox", [])
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            x1 = max(0, min(int(x1 * scale_x), frame_w - 1))
            y1 = max(0, min(int(y1 * scale_y), frame_h - 1))
            x2 = max(0, min(int(x2 * scale_x), frame_w - 1))
            y2 = max(0, min(int(y2 * scale_y), frame_h - 1))

            class_name = obj.get("name", "object")
            confidence = float(obj.get("confidence", 0))
            color = CLASS_COLORS.get(class_name, CLASS_COLORS.get("default", (0, 255, 255)))

            label = f"{class_name}: {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            tw, th = text_size
            cv2.rectangle(frame, (x1, max(y1 - th - 8, 0)), (x1 + tw + 8, y1), color, -1)
            cv2.putText(
                frame, label,
                (x1 + 4, max(y1 - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )


# ─────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────

class VisionGuardGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        init_db()

        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(1360, 820)

        # Стан
        self.cap = None
        self.frame_number = 0
        self.single_image_mode = False
        self.model_frame_size = (640, 640)
        self.fps_counter = 0
        self._last_frame = None  # для zone editor

        # Динамічні налаштування
        self._confidence = DETECTION_CONFIDENCE_THRESHOLD
        self._event_interval = float(MIN_EVENT_INTERVAL)
        self._zone_filter_on = ZONE_FILTER_ENABLED
        self._show_zones = True

        # Зони
        self.zones: list[dict] = load_zones()

        # Вкладки
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.live_tab = QWidget()
        self.events_tab = QWidget()
        self.stats_tab = QWidget()
        self.settings_tab = QWidget()

        self.tabs.addTab(self.live_tab, "📹 Live Video")
        self.tabs.addTab(self.events_tab, "📋 Події")
        self.tabs.addTab(self.stats_tab, "📊 Статистика")
        self.tabs.addTab(self.settings_tab, "⚙️ Налаштування")

        self._build_live_tab()
        self._build_events_tab()
        self._build_stats_tab()
        self._build_settings_tab()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self._update_fps_title)
        self.fps_timer.start(1000)

        self.refresh_event_table()
        self.refresh_stats()

    # ─── Live Tab ───────────────────────────────────────────

    def _build_live_tab(self):
        layout = QVBoxLayout(self.live_tab)

        # Кнопки управління
        ctrl = QHBoxLayout()

        self.btn_camera = QPushButton("▶ Запустити камеру")
        self.btn_camera.clicked.connect(self.start_camera)
        ctrl.addWidget(self.btn_camera)

        self.btn_file = QPushButton("📂 Завантажити відео/кадр")
        self.btn_file.clicked.connect(self.load_file)
        ctrl.addWidget(self.btn_file)

        self.btn_stop = QPushButton("⏹ Зупинити")
        self.btn_stop.clicked.connect(self.stop_stream)
        ctrl.addWidget(self.btn_stop)

        self.btn_zones = QPushButton("🗺 Налаштувати зони")
        self.btn_zones.setStyleSheet("background: #1a6ba0; color: white; font-weight: bold;")
        self.btn_zones.clicked.connect(self.open_zone_editor)
        ctrl.addWidget(self.btn_zones)

        self.chk_show_zones = QCheckBox("Показувати зони")
        self.chk_show_zones.setChecked(True)
        self.chk_show_zones.stateChanged.connect(
            lambda s: setattr(self, "_show_zones", bool(s))
        )
        ctrl.addWidget(self.chk_show_zones)

        layout.addLayout(ctrl)

        # Статус зон
        self.zone_status_label = QLabel()
        self._update_zone_status_label()
        self.zone_status_label.setStyleSheet("color: #4af; font-size: 12px; padding: 2px 4px;")
        layout.addWidget(self.zone_status_label)

        # Відео
        self.video_panel = VideoPanel()
        layout.addWidget(self.video_panel, stretch=4)

        # Лог
        self.live_log = QTextEdit()
        self.live_log.setReadOnly(True)
        self.live_log.setMaximumHeight(140)
        layout.addWidget(self.live_log, stretch=1)

    def _update_zone_status_label(self):
        active = [z for z in self.zones if z.get("enabled", True)]
        if self.zones:
            names = ", ".join(z["name"] for z in active) or "—"
            self.zone_status_label.setText(
                f"Зони ({len(active)}/{len(self.zones)} активних): {names}"
            )
        else:
            self.zone_status_label.setText("Зони не налаштовано — детекція по всьому кадру")

    def append_log(self, message: str):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.live_log.append(f"[{ts}] {message}")
        sb = self.live_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def start_camera(self):
        try:
            self._release()
            self.cap = get_video_capture(0)
            self.frame_number = 0
            self.single_image_mode = False
            self.append_log("Камеру запущено")
            self.timer.start(30)
        except Exception as e:
            self.append_log(f"Помилка підключення камери: {e}")

    def load_file(self):
        try:
            path, _ = QFileDialog.getOpenFileName(
                self, "Відкрити відео або кадр", "",
                "Media Files (*.jpg *.jpeg *.png *.mp4 *.avi *.mkv *.mov)"
            )
            if not path:
                return
            self._release()
            self.cap = get_video_capture(path)
            self.single_image_mode = path.lower().endswith((".jpg", ".jpeg", ".png"))
            self.frame_number = 0
            self.append_log(f"Файл завантажено: {path}")
            self.timer.start(30)
        except Exception as e:
            self.append_log(f"Помилка завантаження файлу: {e}")

    def stop_stream(self):
        self._release()
        self.append_log("Потік зупинено")

    def _release(self):
        self.timer.stop()
        if self.cap is not None and hasattr(self.cap, "release"):
            self.cap.release()
        self.cap = None
        self.single_image_mode = False

    def update_frame(self):
        if self.cap is None:
            return

        frame = read_frame(self.cap)
        if frame is None:
            self.timer.stop()
            self.append_log("Кадр недоступний або потік завершено")
            return

        self._last_frame = frame.copy()

        frame_proc = preprocess_frame(frame, size=self.model_frame_size)
        if frame_proc is None:
            self.timer.stop()
            self.append_log("Кадр не може бути оброблений")
            return

        objects = detect_objects(frame_proc, confidence_threshold=self._confidence)

        events = generate_events(
            objects,
            self.frame_number,
            confidence_threshold=self._confidence,
            event_interval=self._event_interval,
            zones=self.zones if self._zone_filter_on else [],
        )

        self.video_panel.set_frame(
            frame,
            objects,
            zones=self.zones if self._show_zones else None,
            model_size=self.model_frame_size,
        )

        self.fps_counter += 1

        if events:
            for ev in events:
                zone_info = f" | Зона: {ev['zone_name']}" if ev.get("zone_name") else ""
                self.append_log(
                    f"🔔 {ev['event_type']} | {ev['object_class']} "
                    f"| {ev['confidence']:.0%}{zone_info}"
                )
            self.refresh_event_table()
            self.refresh_stats()
        elif self.frame_number % 60 == 0:
            self.append_log(f"Кадр {self.frame_number}: подій не виявлено")

        self.frame_number += 1

        if self.single_image_mode:
            self.timer.stop()

    def _update_fps_title(self):
        self.setWindowTitle(
            f"{APP_NAME} v{APP_VERSION}  |  FPS: {self.fps_counter}"
        )
        self.fps_counter = 0

    def open_zone_editor(self):
        from zone_editor import ZoneEditorDialog
        dlg = ZoneEditorDialog(self, current_frame=self._last_frame)
        dlg.zones_updated.connect(self._on_zones_updated)
        dlg.exec_()

    def _on_zones_updated(self, zones: list[dict]):
        self.zones = zones
        self._update_zone_status_label()
        self.append_log(f"Зони оновлено: {len(zones)} зон збережено")

    # ─── Events Tab ─────────────────────────────────────────

    def _build_events_tab(self):
        layout = QVBoxLayout(self.events_tab)

        btn_row = QHBoxLayout()

        btn_refresh = QPushButton("🔄 Оновити")
        btn_refresh.clicked.connect(self.refresh_event_table)
        btn_row.addWidget(btn_refresh)

        btn_delete = QPushButton("🗑 Видалити вибрану")
        btn_delete.clicked.connect(self._delete_selected_event)
        btn_row.addWidget(btn_delete)

        btn_clear = QPushButton("❌ Очистити всі")
        btn_clear.clicked.connect(self._clear_events)
        btn_row.addWidget(btn_clear)

        layout.addLayout(btn_row)

        self.event_table = QTableWidget()
        self.event_table.setColumnCount(7)
        self.event_table.setHorizontalHeaderLabels([
            "ID", "Час", "Клас", "Тип події", "Впевненість", "Кадр", "Камера"
        ])
        self.event_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.event_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.event_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.event_table.setAlternatingRowColors(True)
        layout.addWidget(self.event_table)

    def refresh_event_table(self):
        events = get_events(limit=300)
        self.event_table.setRowCount(len(events))
        for row, ev in enumerate(events):
            ts = ev.get("timestamp", 0)
            try:
                time_text = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                time_text = ""
            self.event_table.setItem(row, 0, QTableWidgetItem(str(ev.get("id", ""))))
            self.event_table.setItem(row, 1, QTableWidgetItem(time_text))
            self.event_table.setItem(row, 2, QTableWidgetItem(str(ev.get("object_class", ""))))
            self.event_table.setItem(row, 3, QTableWidgetItem(str(ev.get("event_type", ""))))
            self.event_table.setItem(row, 4, QTableWidgetItem(f"{float(ev.get('confidence', 0)):.0%}"))
            self.event_table.setItem(row, 5, QTableWidgetItem(str(ev.get("frame", ""))))
            self.event_table.setItem(row, 6, QTableWidgetItem(str(ev.get("camera_id", ""))))

    def _delete_selected_event(self):
        rows = self.event_table.selectionModel().selectedRows()
        if not rows:
            self.append_log("Виберіть подію для видалення")
            return
        item = self.event_table.item(rows[0].row(), 0)
        if item:
            delete_event(int(item.text()))
            self.refresh_event_table()
            self.refresh_stats()

    def _clear_events(self):
        clear_all_events()
        self.append_log("Усі події очищено")
        self.refresh_event_table()
        self.refresh_stats()

    # ─── Stats Tab ──────────────────────────────────────────

    def _build_stats_tab(self):
        layout = QVBoxLayout(self.stats_tab)

        btn_refresh = QPushButton("🔄 Оновити статистику")
        btn_refresh.clicked.connect(self.refresh_stats)
        layout.addWidget(btn_refresh)

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        font = QFont("Courier New", 11)
        self.stats_text.setFont(font)
        layout.addWidget(self.stats_text)

    def refresh_stats(self):
        stats = get_stats()
        total = stats.get("total", 0)
        last_hour = stats.get("last_hour", 0)
        last_day = stats.get("last_day", 0)
        by_class = stats.get("by_class", {})

        lines = [
            f"{'─' * 40}",
            f"  VisionGuard AI — Статистика подій",
            f"{'─' * 40}",
            f"  Усього подій:         {total}",
            f"  За останню годину:    {last_hour}",
            f"  За останню добу:      {last_day}",
            f"{'─' * 40}",
            "  За класами об'єктів:",
            "",
        ]

        if by_class:
            max_count = max(by_class.values()) or 1
            bar_width = 25
            for cls, cnt in by_class.items():
                bar = "█" * int(cnt / max_count * bar_width)
                lines.append(f"  {cls:<12} {bar:<{bar_width}} {cnt}")
        else:
            lines.append("  — немає даних —")

        lines.append(f"{'─' * 40}")

        # Зони
        lines.append(f"  Активних зон: {len([z for z in self.zones if z.get('enabled', True)])}/{len(self.zones)}")
        if self.zones:
            for z in self.zones:
                status = "✅" if z.get("enabled", True) else "⛔"
                pts = len(z.get("points", []))
                lines.append(f"  {status} {z['name']}  ({pts} точок)")
        lines.append(f"{'─' * 40}")

        self.stats_text.setText("\n".join(lines))

    # ─── Settings Tab ───────────────────────────────────────

    def _build_settings_tab(self):
        layout = QVBoxLayout(self.settings_tab)

        # Детекція
        det_group = QGroupBox("Параметри детекції")
        det_layout = QFormLayout()

        # Confidence slider
        conf_row = QHBoxLayout()
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(5, 95)
        self.slider_conf.setValue(int(self._confidence * 100))
        self.slider_conf.setTickInterval(5)
        self.slider_conf.setTickPosition(QSlider.TicksBelow)
        self.lbl_conf_val = QLabel(f"{self._confidence:.2f}")
        self.slider_conf.valueChanged.connect(self._on_conf_changed)
        conf_row.addWidget(self.slider_conf)
        conf_row.addWidget(self.lbl_conf_val)
        det_layout.addRow("Поріг впевненості:", conf_row)

        # Interval slider
        interval_row = QHBoxLayout()
        self.slider_interval = QSlider(Qt.Horizontal)
        self.slider_interval.setRange(1, 60)
        self.slider_interval.setValue(int(self._event_interval))
        self.slider_interval.setTickInterval(5)
        self.slider_interval.setTickPosition(QSlider.TicksBelow)
        self.lbl_interval_val = QLabel(f"{int(self._event_interval)} с")
        self.slider_interval.valueChanged.connect(self._on_interval_changed)
        interval_row.addWidget(self.slider_interval)
        interval_row.addWidget(self.lbl_interval_val)
        det_layout.addRow("Мін. інтервал подій:", interval_row)

        det_group.setLayout(det_layout)
        layout.addWidget(det_group)

        # Зони
        zone_group = QGroupBox("Налаштування зон")
        zone_layout = QVBoxLayout()

        self.chk_zone_filter = QCheckBox("Фільтрувати події по зонах (тільки об'єкти у зоні)")
        self.chk_zone_filter.setChecked(self._zone_filter_on)
        self.chk_zone_filter.stateChanged.connect(
            lambda s: setattr(self, "_zone_filter_on", bool(s))
        )
        zone_layout.addWidget(self.chk_zone_filter)

        btn_open_editor = QPushButton("🗺 Відкрити редактор зон")
        btn_open_editor.setStyleSheet("background: #1a6ba0; color: white; font-weight: bold; padding: 6px;")
        btn_open_editor.clicked.connect(self.open_zone_editor)
        zone_layout.addWidget(btn_open_editor)

        zone_group.setLayout(zone_layout)
        layout.addWidget(zone_group)

        # Telegram
        tg_group = QGroupBox("Telegram")
        tg_layout = QVBoxLayout()

        self.tg_token = QLineEdit()
        self.tg_token.setPlaceholderText("Bot Token (якщо вже є у config.py — залиш порожнім)")
        self.tg_token.setEchoMode(QLineEdit.Password)
        self.tg_chat = QLineEdit()
        self.tg_chat.setPlaceholderText("Chat ID (якщо вже є у config.py — залиш порожнім)")

        btn_test_tg = QPushButton("📨 Надіслати тестове повідомлення")
        btn_test_tg.clicked.connect(self._send_test_telegram)

        tg_layout.addWidget(self.tg_token)
        tg_layout.addWidget(self.tg_chat)
        tg_layout.addWidget(btn_test_tg)
        tg_group.setLayout(tg_layout)
        layout.addWidget(tg_group)

        layout.addStretch()

    def _on_conf_changed(self, value: int):
        self._confidence = value / 100.0
        self.lbl_conf_val.setText(f"{self._confidence:.2f}")

    def _on_interval_changed(self, value: int):
        self._event_interval = float(value)
        self.lbl_interval_val.setText(f"{value} с")

    def _send_test_telegram(self):
        try:
            ok = send_telegram_message("🔔 Тестове повідомлення від VisionGuard AI")
            self.append_log("Telegram: повідомлення надіслано" if ok else "Telegram: помилка надсилання")
        except Exception as e:
            self.append_log(f"Telegram: {e}")

    def closeEvent(self, event):
        self._release()
        event.accept()


# ─────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = VisionGuardGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()