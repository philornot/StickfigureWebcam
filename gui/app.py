"""
GUI application entry point and styling.

This module provides the main application setup, including
theme configuration and the entry point for GUI mode.
"""

import sys

from PyQt6.QtWidgets import QApplication

from gui.launcher_window import LauncherWindow


def create_app():
    """
    Create and configure the Qt application.

    Returns:
        QApplication: Configured Qt application instance.
    """
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look across platforms

    # Set dark theme
    app.setStyleSheet("""
        QMainWindow, QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QGroupBox {
            border: 1px solid #555;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QPushButton {
            background-color: #0d47a1;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #1565c0;
        }
        QPushButton:pressed {
            background-color: #0a3d91;
        }
        QSlider::groove:horizontal {
            border: 1px solid #555;
            height: 8px;
            background: #3a3a3a;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #0d47a1;
            border: 1px solid #0a3d91;
            width: 18px;
            margin: -5px 0;
            border-radius: 9px;
        }
        QSlider::handle:horizontal:hover {
            background: #1565c0;
        }
        QCheckBox {
            spacing: 8px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 2px solid #555;
            border-radius: 3px;
            background: #3a3a3a;
        }
        QCheckBox::indicator:checked {
            background: #0d47a1;
            border-color: #0d47a1;
        }
        QComboBox {
            background-color: #3a3a3a;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 5px;
        }
        QComboBox:hover {
            border-color: #0d47a1;
        }
        QComboBox::drop-down {
            border: none;
        }
        QComboBox QAbstractItemView {
            background-color: #3a3a3a;
            selection-background-color: #0d47a1;
        }
        QScrollArea {
            border: none;
        }
        QScrollBar:vertical {
            background: #2b2b2b;
            width: 12px;
            border-radius: 6px;
        }
        QScrollBar::handle:vertical {
            background: #555;
            border-radius: 6px;
        }
        QScrollBar::handle:vertical:hover {
            background: #777;
        }
        QTabWidget::pane {
            border: 1px solid #555;
            border-radius: 4px;
        }
        QTabBar::tab {
            background: #3a3a3a;
            color: #ffffff;
            padding: 8px 16px;
            border: 1px solid #555;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background: #0d47a1;
        }
        QTabBar::tab:hover {
            background: #1565c0;
        }
    """)

    return app


def run_gui():
    """
    Run the GUI application.

    Returns:
        int: Application exit code.
    """
    print("=" * 60)
    print("STICKFIGURE WEBCAM - GUI MODE")
    print("=" * 60)

    try:
        app = create_app()
        window = LauncherWindow()
        window.show()
        return app.exec()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(run_gui())
