from pathlib import Path

import pyqtgraph as pg
import qtawesome as qta

from PyQt6.QtGui import QIcon, QFont
from PyQt6.QtWidgets import QSplashScreen, QApplication

from .enum_state import StartUpState


def start_up_appliciation_theme(splash_screen: QSplashScreen, app: QApplication, assets: Path) -> StartUpState:
    splash_screen.showMessage("Setting application theme...")
    
    # application theme
    # qta.dark(app)
    qta.light(app)

    # set window icon
    icon_path = assets / "icons" / "icon_color.svg"
    app.setWindowIcon(QIcon(icon_path.as_posix()))

    # set default font
    splash_screen.showMessage("Load default font...")
    font = QFont()
    font.setFamily(font.defaultFamily())
    font.setWeight(QFont.Weight.ExtraLight)
    font.setPointSize(9)
    app.setFont(font)

    # even though it is pyqtgraph we it is part of the theme :^)
    pg.setConfigOption(opt="background", value=pg.mkColor(255, 255, 255))

    splash_screen.showMessage("Application theme set...")
    return StartUpState.SUCCESS
