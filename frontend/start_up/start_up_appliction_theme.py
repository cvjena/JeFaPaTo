import pathlib

import pyqtgraph as pg
import qtawesome as qta
from qtpy import QtGui, QtWidgets

from .enum_state import StartUpState


def start_up_appliciation_theme(
    splash_screen: QtWidgets.QSplashScreen, **kwargs
) -> StartUpState:
    splash_screen.showMessage("Setting application theme...")
    app = kwargs.get("app")
    # qta.dark(app)
    qta.light(app)
    app.setWindowIcon(
        QtGui.QIcon(
            (
                pathlib.Path(__file__).parent.parent.parent
                / "assets"
                / "icon_256x256.png"
            ).as_posix()
        )
    )

    # even though it is pyqtgraph we it is part of the theme :^)
    pg.setConfigOption(opt="background", value=pg.mkColor(255, 255, 255))

    splash_screen.showMessage("Application theme set...")
    return StartUpState.SUCCESS