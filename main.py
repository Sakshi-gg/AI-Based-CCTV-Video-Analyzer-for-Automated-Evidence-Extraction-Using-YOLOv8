import sys
from PySide6.QtWidgets import QApplication
from gui import AnalyzerWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AnalyzerWindow()
    window.show()
    sys.exit(app.exec())

