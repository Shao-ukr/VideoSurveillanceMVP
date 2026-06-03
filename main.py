# main.py
import sys
from PyQt5.QtWidgets import QApplication
from gui import VisionGuardGUI  # <-- тут правильна назва класу

def main():
    app = QApplication(sys.argv)
    window = VisionGuardGUI()  # <-- замінити VideoSurveillanceGUI на VisionGuardGUI
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()