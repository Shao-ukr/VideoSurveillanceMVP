# gui.py — VisionGuard AI v2.0
import sys
import cv2
import datetime
import os

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout,
    QHBoxLayout, QLabel, QFileDialog, QTextEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QTabWidget, QLineEdit, QGroupBox,
    QFormLayout, QSlider, QCheckBox, QSizePolicy, QComboBox,
    QMessageBox, QSplitter, QScrollArea,
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

from camera import get_video_capture, read_frame
from preprocessing import preprocess_frame
from detector import detect_objects
from events import generate_events
from db import init_db, get_events, get_stats, delete_event, clear_all_events, export_to_csv, export_to_excel
from notifications import send_telegram_message_sync
from zones import load_zones, draw_zones_on_frame, ZONE_TYPES

try:
    from config import APP_NAME, APP_VERSION
except ImportError:
    APP_NAME, APP_VERSION = "VisionGuard AI", "2.0"

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


# ─────────────────────────────────────────────────────────────────────────────
# Worker Thread — обробка кадрів у фоновому потоці
# ─────────────────────────────────────────────────────────────────────────────

class FrameWorker(QThread):
    """QThread для обробки кадрів без блокування GUI."""
    result_ready = pyqtSignal(object, list, list)  # frame, objects, events
    error_occurred = pyqtSignal(str)
    finished_stream = pyqtSignal()

    def __init__(self, cap, zones, confidence, event_interval,
                 zone_filter_on, model_frame_size, parent=None):
        super().__init__(parent)
        self.cap = cap
        self.zones = zones
        self.confidence = confidence
        self.event_interval = event_interval
        self.zone_filter_on = zone_filter_on
        self.model_frame_size = model_frame_size
        self._running = True
        self._frame_number = 0

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            frame = read_frame(self.cap)
            if frame is None:
                self.finished_stream.emit()
                break

            raw_frame = frame.copy()
            frame_proc = preprocess_frame(frame, size=self.model_frame_size)
            if frame_proc is None:
                continue

            objects = detect_objects(frame_proc, confidence_threshold=self.confidence)

            events = generate_events(
                objects,
                self._frame_number,
                confidence_threshold=self.confidence,
                event_interval=self.event_interval,
                zones=self.zones if self.zone_filter_on else [],
                raw_frame=raw_frame,
            )

            self.result_ready.emit(raw_frame, objects, events)
            self._frame_number += 1
            self.msleep(30)  # ~33 fps max


# ─────────────────────────────────────────────────────────────────────────────
# VideoPanel
# ─────────────────────────────────────────────────────────────────────────────

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
                color: #555;
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
        active_zone_names: set = None,
    ):
        if frame_bgr is None or frame_bgr.size == 0:
            return

        frame_draw = frame_bgr.copy()
        h, w = frame_draw.shape[:2]

        if show_zones and zones:
            frame_draw = draw_zones_on_frame(
                frame_draw, zones, w, h,
                original_w=model_size[0],
                original_h=model_size[1],
                active_zone_names=active_zone_names or set(),
            )

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


# ─────────────────────────────────────────────────────────────────────────────
# AlertHistoryPanel — thumbnails подій
# ─────────────────────────────────────────────────────────────────────────────

class AlertHistoryPanel(QWidget):
    """Панель мініатюр кадрів подій."""
    event_frame_clicked = pyqtSignal(str)  # path

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._container = QWidget()
        self._grid = QHBoxLayout(self._container)
        self._grid.setAlignment(Qt.AlignLeft)
        scroll.setWidget(self._container)
        layout.addWidget(scroll)

        self._thumb_labels: list[QLabel] = []

    def add_event_frame(self, frame_path: str, label_text: str):
        if not frame_path or not os.path.exists(frame_path):
            return

        frame = cv2.imread(frame_path)
        if frame is None:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        q_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(160, 120, Qt.KeepAspectRatio)

        container = QWidget()
        v = QVBoxLayout(container)
        v.setContentsMargins(4, 4, 4, 4)

        lbl_img = QLabel()
        lbl_img.setPixmap(pixmap)
        lbl_img.setCursor(Qt.PointingHandCursor)
        lbl_img.setToolTip(f"Клік — відкрити кадр\n{label_text}")

        # Клік по мініатюрі
        path_copy = frame_path
        lbl_img.mousePressEvent = lambda e, p=path_copy: self.event_frame_clicked.emit(p)

        lbl_text = QLabel(label_text[:20])
        lbl_text.setAlignment(Qt.AlignCenter)
        lbl_text.setStyleSheet("font-size: 10px; color: #aaa;")

        v.addWidget(lbl_img)
        v.addWidget(lbl_text)

        self._grid.addWidget(container)
        self._thumb_labels.append(lbl_img)

        # Обмежуємо кількість мініатюр
        if len(self._thumb_labels) > 30:
            oldest = self._grid.itemAt(0).widget()
            self._grid.removeWidget(oldest)
            oldest.deleteLater()
            self._thumb_labels.pop(0)


# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────

class VisionGuardGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        init_db()

        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(1440, 880)

        # Стан
        self._worker: FrameWorker | None = None
        self._cap = None
        self._single_image_mode = False
        self._model_frame_size = (640, 640)
        self._fps_counter = 0
        self._last_frame = None
        self._active_zone_names: set = set()

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

        self.live_tab    = QWidget()
        self.events_tab  = QWidget()
        self.stats_tab   = QWidget()
        self.history_tab = QWidget()
        self.settings_tab = QWidget()

        self.tabs.addTab(self.live_tab,     "📹 Live Video")
        self.tabs.addTab(self.events_tab,   "📋 Події")
        self.tabs.addTab(self.stats_tab,    "📊 Статистика")
        self.tabs.addTab(self.history_tab,  "🖼 Кадри подій")
        self.tabs.addTab(self.settings_tab, "⚙️ Налаштування")

        self._build_live_tab()
        self._build_events_tab()
        self._build_stats_tab()
        self._build_history_tab()
        self._build_settings_tab()

        self._fps_timer = QTimer()
        self._fps_timer.timeout.connect(self._update_fps_title)
        self._fps_timer.start(1000)

        self.refresh_event_table()
        self.refresh_stats()

        from PyQt5.QtGui import QIcon

        self.setWindowIcon(QIcon("logo.png"))  # шлях до логотипу у проекті

    # ── Live Tab ─────────────────────────────────────────────────────────────

    def _build_live_tab(self):
        layout = QVBoxLayout(self.live_tab)

        # Кнопки
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
        self.btn_zones.setStyleSheet("background:#1a6ba0; color:white; font-weight:bold;")
        self.btn_zones.clicked.connect(self.open_zone_editor)
        ctrl.addWidget(self.btn_zones)

        self.chk_show_zones = QCheckBox("Показувати зони")
        self.chk_show_zones.setChecked(True)
        self.chk_show_zones.stateChanged.connect(
            lambda s: setattr(self, "_show_zones", bool(s))
        )
        ctrl.addWidget(self.chk_show_zones)

        self.chk_zone_filter_live = QCheckBox("Фільтр зон")
        self.chk_zone_filter_live.setChecked(self._zone_filter_on)
        self.chk_zone_filter_live.stateChanged.connect(
            lambda s: setattr(self, "_zone_filter_on", bool(s))
        )
        ctrl.addWidget(self.chk_zone_filter_live)

        layout.addLayout(ctrl)

        # Статус зон
        self.zone_status_label = QLabel()
        self._update_zone_status_label()
        self.zone_status_label.setStyleSheet("color:#4af; font-size:12px; padding:2px 4px;")
        layout.addWidget(self.zone_status_label)

        # Відео + лог
        splitter = QSplitter(Qt.Vertical)

        self.video_panel = VideoPanel()
        splitter.addWidget(self.video_panel)

        self.live_log = QTextEdit()
        self.live_log.setReadOnly(True)
        self.live_log.setMaximumHeight(130)
        splitter.addWidget(self.live_log)

        splitter.setSizes([600, 130])
        layout.addWidget(splitter)

    def _update_zone_status_label(self):
        active = [z for z in self.zones if z.get("enabled", True)]
        if self.zones:
            parts = []
            for z in active:
                zone_type = z.get("zone_type", "secondary")
                type_label = ZONE_TYPES.get(zone_type, {}).get("label", "")
                parts.append(f"{z['name']} [{type_label}]")
            names = ", ".join(parts) or "—"
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
            self._stop_worker()
            self._cap = get_video_capture(0)
            self._single_image_mode = False
            self.append_log("Камеру запущено")
            self._start_worker()
        except Exception as e:
            self.append_log(f"Помилка: {e}")

    def load_file(self):
        try:
            path, _ = QFileDialog.getOpenFileName(
                self, "Відкрити відео або кадр", "",
                "Media Files (*.jpg *.jpeg *.png *.mp4 *.avi *.mkv *.mov)"
            )
            if not path:
                return
            self._stop_worker()
            self._cap = get_video_capture(path)
            self._single_image_mode = path.lower().endswith((".jpg", ".jpeg", ".png"))
            self.append_log(f"Файл: {path}")
            self._start_worker()
        except Exception as e:
            self.append_log(f"Помилка: {e}")

    def stop_stream(self):
        self._stop_worker()
        self.append_log("Потік зупинено")

    def _start_worker(self):
        if self._cap is None:
            return
        self._worker = FrameWorker(
            cap=self._cap,
            zones=self.zones,
            confidence=self._confidence,
            event_interval=self._event_interval,
            zone_filter_on=self._zone_filter_on,
            model_frame_size=self._model_frame_size,
        )
        self._worker.result_ready.connect(self._on_frame_result)
        self._worker.finished_stream.connect(lambda: self.append_log("Потік завершено"))
        self._worker.error_occurred.connect(self.append_log)
        self._worker.start()

    def _stop_worker(self):
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait(500)
            self._worker = None
        if self._cap is not None and hasattr(self._cap, "release"):
            self._cap.release()
        self._cap = None

    def _on_frame_result(self, frame, objects, events):
        self._last_frame = frame

        # Оновлюємо активні зони
        self._active_zone_names = {ev.get("zone_name") for ev in events if ev.get("zone_name")}

        self.video_panel.set_frame(
            frame,
            objects,
            zones=self.zones if self._show_zones else None,
            model_size=self._model_frame_size,
            show_zones=self._show_zones,
            active_zone_names=self._active_zone_names,
        )
        self._fps_counter += 1

        if events:
            for ev in events:
                zone_info = f" | Зона: {ev['zone_name']}" if ev.get("zone_name") else ""
                self.append_log(
                    f"🔔 {ev['event_type']} | {ev['object_class']} "
                    f"| {ev['confidence']:.0%}{zone_info}"
                )
                # Додаємо мініатюру
                if ev.get("frame_path"):
                    self.alert_panel.add_event_frame(
                        ev["frame_path"],
                        f"{ev['object_class']} {ev['confidence']:.0%}"
                    )
            self.refresh_event_table()
            self.refresh_stats()

    def _update_fps_title(self):
        self.setWindowTitle(
            f"{APP_NAME} v{APP_VERSION}  |  FPS: {self._fps_counter}"
        )
        self._fps_counter = 0

    def open_zone_editor(self):
        from zone_editor import ZoneEditorDialog
        dlg = ZoneEditorDialog(self, current_frame=self._last_frame)
        dlg.zones_updated.connect(self._on_zones_updated)
        dlg.exec_()

    def _on_zones_updated(self, zones: list[dict]):
        self.zones = zones
        if self._worker:
            self._worker.zones = zones
        self._update_zone_status_label()
        self.append_log(f"Зони оновлено: {len(zones)}")

    # ── Events Tab ────────────────────────────────────────────────────────────

    def _build_events_tab(self):
        layout = QVBoxLayout(self.events_tab)

        # Фільтри
        filter_group = QGroupBox("Фільтри")
        filter_layout = QHBoxLayout()

        filter_layout.addWidget(QLabel("Клас:"))
        self.filter_class = QComboBox()
        self.filter_class.addItem("Усі", None)
        for cls in ["person", "car", "bicycle", "truck", "motorcycle"]:
            self.filter_class.addItem(cls, cls)
        filter_layout.addWidget(self.filter_class)

        filter_layout.addWidget(QLabel("Тип події:"))
        self.filter_event_type = QLineEdit()
        self.filter_event_type.setPlaceholderText("пошук...")
        self.filter_event_type.setMaximumWidth(180)
        filter_layout.addWidget(self.filter_event_type)

        filter_layout.addWidget(QLabel("Мін. впевненість:"))
        self.filter_conf = QSlider(Qt.Horizontal)
        self.filter_conf.setRange(0, 95)
        self.filter_conf.setValue(0)
        self.filter_conf.setMaximumWidth(120)
        self.lbl_filter_conf = QLabel("0%")
        self.filter_conf.valueChanged.connect(
            lambda v: self.lbl_filter_conf.setText(f"{v}%")
        )
        filter_layout.addWidget(self.filter_conf)
        filter_layout.addWidget(self.lbl_filter_conf)

        btn_apply = QPushButton("🔍 Застосувати")
        btn_apply.clicked.connect(self.refresh_event_table)
        filter_layout.addWidget(btn_apply)

        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)

        # Кнопки
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

        btn_export_csv = QPushButton("📄 Експорт CSV")
        btn_export_csv.clicked.connect(self._export_csv)
        btn_row.addWidget(btn_export_csv)

        btn_export_xlsx = QPushButton("📊 Експорт Excel")
        btn_export_xlsx.clicked.connect(self._export_excel)
        btn_row.addWidget(btn_export_xlsx)

        layout.addLayout(btn_row)

        # Таблиця
        self.event_table = QTableWidget()
        self.event_table.setColumnCount(8)
        self.event_table.setHorizontalHeaderLabels([
            "ID", "Час", "Клас", "Тип події", "Впевненість", "Кадр", "Камера", "Кадр файл"
        ])
        self.event_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.event_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.event_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.event_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.event_table.setAlternatingRowColors(True)
        self.event_table.doubleClicked.connect(self._on_event_double_click)
        layout.addWidget(self.event_table)

    def refresh_event_table(self):
        cls_filter = self.filter_class.currentData() if hasattr(self, "filter_class") else None
        type_filter = self.filter_event_type.text().strip() if hasattr(self, "filter_event_type") else None
        min_conf = (self.filter_conf.value() / 100.0) if hasattr(self, "filter_conf") else 0.0

        events = get_events(
            limit=500,
            object_class_filter=cls_filter,
            event_type_filter=type_filter or None,
            min_confidence=min_conf,
        )
        self.event_table.setRowCount(len(events))
        for row, ev in enumerate(events):
            ts = ev.get("timestamp", 0)
            try:
                time_text = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                time_text = ""

            conf = float(ev.get("confidence", 0))
            conf_item = QTableWidgetItem(f"{conf:.0%}")
            if conf >= 0.8:
                conf_item.setBackground(QColor("#c6efce"))
            elif conf >= 0.5:
                conf_item.setBackground(QColor("#ffeb9c"))
            else:
                conf_item.setBackground(QColor("#ffc7ce"))

            self.event_table.setItem(row, 0, QTableWidgetItem(str(ev.get("id", ""))))
            self.event_table.setItem(row, 1, QTableWidgetItem(time_text))
            self.event_table.setItem(row, 2, QTableWidgetItem(str(ev.get("object_class", ""))))
            self.event_table.setItem(row, 3, QTableWidgetItem(str(ev.get("event_type", ""))))
            self.event_table.setItem(row, 4, conf_item)
            self.event_table.setItem(row, 5, QTableWidgetItem(str(ev.get("frame", ""))))
            self.event_table.setItem(row, 6, QTableWidgetItem(str(ev.get("camera_id", ""))))
            frame_path = ev.get("frame_path") or ""
            self.event_table.setItem(row, 7, QTableWidgetItem(os.path.basename(frame_path) if frame_path else "—"))

    def _on_event_double_click(self, index):
        """При подвійному кліку показує кадр події у Live Video."""
        row = index.row()
        path_item = self.event_table.item(row, 7)
        if path_item and path_item.text() != "—":
            events_dir = "events"
            full_path = os.path.join(events_dir, path_item.text())
            if os.path.exists(full_path):
                frame = cv2.imread(full_path)
                if frame is not None:
                    self.video_panel.set_frame(frame)
                    self.tabs.setCurrentIndex(0)

    def _delete_selected_event(self):
        rows = self.event_table.selectionModel().selectedRows()
        if not rows:
            return
        item = self.event_table.item(rows[0].row(), 0)
        if item:
            delete_event(int(item.text()))
            self.refresh_event_table()
            self.refresh_stats()

    def _clear_events(self):
        reply = QMessageBox.question(
            self, "Підтвердження", "Очистити всі події?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            clear_all_events()
            self.append_log("Усі події очищено")
            self.refresh_event_table()
            self.refresh_stats()

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Зберегти CSV", "events_export.csv", "CSV (*.csv)")
        if path:
            count = export_to_csv(path)
            self.append_log(f"CSV: {count} подій → {path}")

    def _export_excel(self):
        path, _ = QFileDialog.getSaveFileName(self, "Зберегти Excel", "events_export.xlsx", "Excel (*.xlsx)")
        if path:
            try:
                count = export_to_excel(path)
                self.append_log(f"Excel: {count} подій → {path}")
            except ImportError as e:
                QMessageBox.warning(self, "Помилка", str(e))

    # ── Stats Tab ─────────────────────────────────────────────────────────────

    def _build_stats_tab(self):
        layout = QVBoxLayout(self.stats_tab)

        btn_refresh = QPushButton("🔄 Оновити статистику")
        btn_refresh.clicked.connect(self.refresh_stats)
        layout.addWidget(btn_refresh)

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setFont(QFont("Courier New", 11))
        layout.addWidget(self.stats_text)

    def refresh_stats(self):
        stats = get_stats()
        total = stats.get("total", 0)
        last_hour = stats.get("last_hour", 0)
        last_day = stats.get("last_day", 0)
        by_class = stats.get("by_class", {})
        by_hour = stats.get("by_hour", {})

        sep = "─" * 42
        lines = [
            sep,
            f"  VisionGuard AI v{APP_VERSION} — Статистика",
            sep,
            f"  Усього подій:         {total}",
            f"  За останню годину:    {last_hour}",
            f"  За останню добу:      {last_day}",
            sep,
            "  За класами об'єктів:",
            "",
        ]

        if by_class:
            max_count = max(by_class.values()) or 1
            bar_width = 25
            for cls, cnt in by_class.items():
                bar = "█" * int(cnt / max_count * bar_width)
                pct = cnt / total * 100 if total else 0
                lines.append(f"  {cls:<12} {bar:<{bar_width}} {cnt:>4}  ({pct:.1f}%)")
        else:
            lines.append("  — немає даних —")

        lines.append(sep)

        # Активність по годинах
        if by_hour:
            lines.append("  Активність по годинах (24 год):")
            max_h = max(by_hour.values()) or 1
            for h in sorted(by_hour.keys()):
                bar = "█" * int(by_hour[h] / max_h * 20)
                lines.append(f"  {h}:00  {bar:<20} {by_hour[h]}")
            lines.append(sep)

        # Зони
        active_zones = [z for z in self.zones if z.get("enabled", True)]
        lines.append(f"  Активних зон: {len(active_zones)}/{len(self.zones)}")
        for z in self.zones:
            status = "✅" if z.get("enabled", True) else "⛔"
            zone_type = z.get("zone_type", "secondary")
            type_label = ZONE_TYPES.get(zone_type, {}).get("label", "")
            pts = len(z.get("points", []))
            lines.append(f"  {status} {z['name']}  [{type_label}]  ({pts} точок)")
        lines.append(sep)

        self.stats_text.setText("\n".join(lines))

    # ── History Tab ──────────────────────────────────────────────────────────

    def _build_history_tab(self):
        layout = QVBoxLayout(self.history_tab)

        btn_row = QHBoxLayout()
        btn_reload = QPushButton("🔄 Завантажити кадри з папки events/")
        btn_reload.clicked.connect(self._load_event_frames)
        btn_row.addWidget(btn_reload)
        layout.addLayout(btn_row)

        self.alert_panel = AlertHistoryPanel()
        self.alert_panel.event_frame_clicked.connect(self._show_event_frame)
        layout.addWidget(self.alert_panel)

    def _load_event_frames(self):
        """Завантажує всі збережені кадри з папки events/."""
        events_dir = "events"
        if not os.path.exists(events_dir):
            return
        for fname in sorted(os.listdir(events_dir))[-30:]:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(events_dir, fname)
                parts = fname.replace(".jpg", "").split("_")
                label = "_".join(parts[2:4]) if len(parts) >= 4 else fname[:15]
                self.alert_panel.add_event_frame(path, label)

    def _show_event_frame(self, path: str):
        """Показує кадр події у вкладці Live Video."""
        frame = cv2.imread(path)
        if frame is not None:
            self.video_panel.set_frame(frame)
            self.tabs.setCurrentIndex(0)

    # ── Settings Tab ──────────────────────────────────────────────────────────

    def _build_settings_tab(self):
        layout = QVBoxLayout(self.settings_tab)

        # Детекція
        det_group = QGroupBox("Параметри детекції")
        det_layout = QFormLayout()

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

        self.chk_zone_filter = QCheckBox("Фільтрувати події по зонах")
        self.chk_zone_filter.setChecked(self._zone_filter_on)
        self.chk_zone_filter.stateChanged.connect(
            lambda s: setattr(self, "_zone_filter_on", bool(s))
        )
        zone_layout.addWidget(self.chk_zone_filter)

        btn_open_editor = QPushButton("🗺 Відкрити редактор зон")
        btn_open_editor.setStyleSheet("background:#1a6ba0; color:white; font-weight:bold; padding:6px;")
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
        self.tg_chat.setPlaceholderText("Chat ID")

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
        if self._worker:
            self._worker.confidence = self._confidence

    def _on_interval_changed(self, value: int):
        self._event_interval = float(value)
        self.lbl_interval_val.setText(f"{value} с")
        if self._worker:
            self._worker.event_interval = self._event_interval

    def _send_test_telegram(self):
        ok = send_telegram_message_sync("🔔 Тестове повідомлення від VisionGuard AI")
        self.append_log("Telegram: OK" if ok else "Telegram: помилка")

    def closeEvent(self, event):
        self._stop_worker()
        event.accept()


# ─────────────────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = VisionGuardGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()