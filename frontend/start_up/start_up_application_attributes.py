from PyQt6.QtWidgets import QSplashScreen, QApplication
from PyQt6.QtCore import Qt

from .enum_state import StartUpState


def start_up_appliciation_attributes(splash_screen: QSplashScreen, app: QApplication, **kwargs) -> StartUpState:
    splash_screen.showMessage("Setting application attributes...")

    if hasattr(Qt.ApplicationAttribute, "AA_UseHighDpiPixmaps"):
        app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)

    app.setApplicationDisplayName("JeFaPaTo")
    app.setApplicationName("JeFaPaTo")

    splash_screen.showMessage("Application attributes set...")
    return StartUpState.SUCCESS
