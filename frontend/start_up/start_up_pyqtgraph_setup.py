import pyqtgraph as pg
from qtpy import QtWidgets

from .enum_state import StartUpState


def start_up_pytqgraph_setup(
    splash_screen: QtWidgets.QSplashScreen, **_
) -> StartUpState:
    splash_screen.showMessage("Setting pyqtgraph attributes...")

    # the images have to be flipped as the default is upside down
    pg.setConfigOption(opt="imageAxisOrder", value="row-major")

    splash_screen.showMessage("pyqtgraph attributes set...")
    return StartUpState.SUCCESS
