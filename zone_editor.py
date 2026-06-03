# zone_editor.py — Діалог малювання зон на відео-кадрі
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QWidget,
    QCheckBox,
    QSplitter,
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QPolygon
from PyQt5.QtCore import Qt, QPoint, pyqtSignal

from zones import load_zones, save_zones


class ZoneCanvas(QLabel):
    """
    Канвас для малювання полігон-зони мишею.
    Клік лівою — додає точку.
    Клік правою — закриває полігон.
    """
    zone_finished = pyqtSignal(list)  # список точок [[x,y], ...]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.CrossCursor)

        self._base_pixmap: QPixmap | None = None
        self._points: list[QPoint] = []
        self._drawing = False

    def set_frame(self, frame_bgr: np.ndarray):
        """Встановлює базовий кадр як фон полотна."""
        if frame_bgr is None or frame_bgr.size == 0:
            return
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        q_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self._base_pixmap = QPixmap.fromImage(q_img).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._points = []
        self._drawing = False
        self._redraw()

    def start_drawing(self):
        self._points = []
        self._drawing = True
        self._redraw()

    def clear_points(self):
        self._points = []
        self._drawing = False
        self._redraw()

    def mousePressEvent(self, event):
        if not self._drawing:
            return
        if event.button() == Qt.LeftButton:
            self._points.append(event.pos())
            self._redraw()
        elif event.button() == Qt.RightButton:
            self._finish_polygon()

    def _finish_polygon(self):
        if len(self._points) < 3:
            return

        # Конвертуємо точки канвасу у координати оригінального кадру (640x640)
        if self._base_pixmap is None:
            return

        pm_w = self._base_pixmap.width()
        pm_h = self._base_pixmap.height()

        # offset — де пікселей починається зображення всередині QLabel
        label_w = self.width()
        label_h = self.height()
        offset_x = (label_w - pm_w) // 2
        offset_y = (label_h - pm_h) // 2

        points_norm = []
        for p in self._points:
            rel_x = p.x() - offset_x
            rel_y = p.y() - offset_y
            # Нормалізуємо до 640x640
            norm_x = int(rel_x * 640 / pm_w)
            norm_y = int(rel_y * 640 / pm_h)
            norm_x = max(0, min(norm_x, 640))
            norm_y = max(0, min(norm_y, 640))
            points_norm.append([norm_x, norm_y])

        self._drawing = False
        self.zone_finished.emit(points_norm)
        self._redraw()

    def _redraw(self):
        if self._base_pixmap is None:
            self.setText("Немає кадру. Запустіть камеру або відкрийте файл.")
            return

        result = self._base_pixmap.copy()
        painter = QPainter(result)

        if self._points:
            pen = QPen(QColor(0, 220, 255), 2, Qt.SolidLine)
            painter.setPen(pen)

            # Лінії між точками
            for i in range(len(self._points) - 1):
                painter.drawLine(self._points[i], self._points[i + 1])

            # Точки
            painter.setBrush(QColor(0, 220, 255))
            for p in self._points:
                painter.drawEllipse(p, 5, 5)

            # Замикаюча лінія (пунктир)
            if len(self._points) >= 3 and self._drawing:
                pen.setStyle(Qt.DashLine)
                pen.setColor(QColor(255, 200, 0))
                painter.setPen(pen)
                painter.drawLine(self._points[-1], self._points[0])

        painter.end()
        self.setPixmap(result)


class ZoneEditorDialog(QDialog):
    """
    Повноцінний діалог керування зонами.
    - Список існуючих зон зліва
    - Канвас для малювання справа
    - Кнопки: Нова зона, Видалити, Зберегти
    """

    zones_updated = pyqtSignal(list)

    def __init__(self, parent=None, current_frame: np.ndarray = None):
        super().__init__(parent)
        self.setWindowTitle("Редактор зон виявлення")
        self.resize(1000, 600)
        self.setModal(True)

        self._zones = load_zones()
        self._current_frame = current_frame
        self._pending_points: list[list[int]] = []

        self._build_ui()
        self._refresh_list()

        if current_frame is not None:
            self.canvas.set_frame(current_frame)

    def _build_ui(self):
        main = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # ── Ліва панель (список зон + кнопки) ──
        left = QWidget()
        left_layout = QVBoxLayout(left)

        self.zone_list = QListWidget()
        self.zone_list.currentRowChanged.connect(self._on_zone_selected)
        left_layout.addWidget(QLabel("Зони:"))
        left_layout.addWidget(self.zone_list)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Назва:"))
        self.zone_name_edit = QLineEdit()
        self.zone_name_edit.setPlaceholderText("Наприклад: Вхід")
        name_row.addWidget(self.zone_name_edit)
        left_layout.addLayout(name_row)

        self.zone_enabled_cb = QCheckBox("Увімкнена")
        self.zone_enabled_cb.setChecked(True)
        left_layout.addWidget(self.zone_enabled_cb)

        btn_new = QPushButton("🖊 Нова зона (малюй мишею)")
        btn_new.clicked.connect(self._start_new_zone)
        left_layout.addWidget(btn_new)

        btn_clear = QPushButton("🗑 Очистити поточне малювання")
        btn_clear.clicked.connect(self.canvas.clear_points if hasattr(self, "canvas") else lambda: None)
        left_layout.addWidget(btn_clear)

        btn_delete = QPushButton("❌ Видалити вибрану зону")
        btn_delete.clicked.connect(self._delete_zone)
        left_layout.addWidget(btn_delete)

        hint = QLabel(
            "💡 Ліва кнопка миші — додати точку\n"
            "Права кнопка миші — завершити зону\n"
            "(мінімум 3 точки)"
        )
        hint.setStyleSheet("color: #888; font-size: 11px;")
        left_layout.addWidget(hint)

        btn_save = QPushButton("💾 Зберегти всі зони")
        btn_save.setStyleSheet("background: #2a7; color: white; font-weight: bold; padding: 6px;")
        btn_save.clicked.connect(self._save_and_close)
        left_layout.addWidget(btn_save)

        btn_cancel = QPushButton("Скасувати")
        btn_cancel.clicked.connect(self.reject)
        left_layout.addWidget(btn_cancel)

        splitter.addWidget(left)

        # ── Права панель (канвас) ──
        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.canvas = ZoneCanvas()
        self.canvas.zone_finished.connect(self._on_zone_drawn)
        right_layout.addWidget(self.canvas)

        # Тепер прив'язуємо btn_clear до реального canvas
        btn_clear.clicked.disconnect()
        btn_clear.clicked.connect(self.canvas.clear_points)

        splitter.addWidget(right)
        splitter.setSizes([280, 720])

        main.addWidget(splitter)

        if self._current_frame is not None:
            self.canvas.set_frame(self._current_frame)

    def _refresh_list(self):
        self.zone_list.clear()
        for zone in self._zones:
            status = "✅" if zone.get("enabled", True) else "⛔"
            pts = len(zone.get("points", []))
            item = QListWidgetItem(f"{status} {zone['name']}  ({pts} точок)")
            self.zone_list.addItem(item)

    def _on_zone_selected(self, row: int):
        if 0 <= row < len(self._zones):
            zone = self._zones[row]
            self.zone_name_edit.setText(zone.get("name", ""))
            self.zone_enabled_cb.setChecked(zone.get("enabled", True))
            # Відобразити зону на кадрі
            self._draw_zone_preview(zone)

    def _draw_zone_preview(self, zone: dict):
        if self._current_frame is None:
            return
        frame_copy = self._current_frame.copy()
        from zones import draw_zones_on_frame
        h, w = frame_copy.shape[:2]
        frame_copy = draw_zones_on_frame(frame_copy, [zone], w, h)
        self.canvas.set_frame(frame_copy)

    def _start_new_zone(self):
        name = self.zone_name_edit.text().strip() or f"Зона {len(self._zones)+1}"
        self._pending_name = name
        self._pending_enabled = self.zone_enabled_cb.isChecked()
        self.canvas.start_drawing()

    def _on_zone_drawn(self, points: list[list[int]]):
        name = getattr(self, "_pending_name", f"Зона {len(self._zones)+1}")
        enabled = getattr(self, "_pending_enabled", True)
        new_zone = {"name": name, "points": points, "enabled": enabled}
        self._zones.append(new_zone)
        self._refresh_list()
        self.zone_list.setCurrentRow(len(self._zones) - 1)

    def _delete_zone(self):
        row = self.zone_list.currentRow()
        if row < 0 or row >= len(self._zones):
            return
        name = self._zones[row]["name"]
        reply = QMessageBox.question(
            self,
            "Підтвердження",
            f"Видалити зону «{name}»?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._zones.pop(row)
            self._refresh_list()
            if self._current_frame is not None:
                self.canvas.set_frame(self._current_frame)

    def _save_and_close(self):
        save_zones(self._zones)
        self.zones_updated.emit(self._zones)
        self.accept()