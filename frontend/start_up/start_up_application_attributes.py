from qtpy import QtCore, QtWidgets

from .enum_state import StartUpState


def start_up_appliciation_attributes(
    splash_screen: QtWidgets.QSplashScreen, **kwargs
) -> StartUpState:
    splash_screen.showMessage("Setting application attributes...")

    app = kwargs.get("app")
    if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
        app.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)

    splash_screen.showMessage("Application attributes set...")
    return StartUpState.SUCCESS
