__all__ = ["StartUpSplashScreen"]

import inspect
import time
import typing
from importlib import util
from pathlib import Path

import structlog
from PyQt6 import QtGui
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QSplashScreen

from . import enum_state

logger = structlog.get_logger()

ASSETS_PATH = Path(__file__).parent.parent / "assets"

class StartUpException(Exception):
    pass

class StartUpSplashScreen(QSplashScreen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)

        # show on screen 1
        screen = QApplication.screens()[0]
        self.setGeometry(screen.geometry())

        path = ASSETS_PATH / "icons" / "icon_color.svg"
        self.setPixmap(QtGui.QPixmap(path.as_posix()))
        self.setEnabled(False)
        self.start_up_tasks: list[typing.Callable] = []

        for files in Path(__file__).parent.glob("start_up_*.py"):
            # this feels really hacky, but it works
            # inspired by the way pytest loads tests
            module = files.with_suffix("").relative_to(Path(__file__).parent.parent.parent)
            module = ".".join(module.parts)

            spec = util.spec_from_file_location(module, files)
            if spec is None:
                raise StartUpException(f"Could not load {files}")

            mod = util.module_from_spec(spec)
            
            assert spec.loader is not None
            spec.loader.exec_module(mod)

            functions = inspect.getmembers(mod, inspect.isfunction)
            for f_name, f_value in functions:
                if f_name.startswith("start_up_"):
                    sig = inspect.signature(f_value)
                    if "splash_screen" not in sig.parameters:
                        raise StartUpException(f"{f_name} must have a parameter named 'splash_screen'")

                    if sig.parameters["splash_screen"].annotation != QSplashScreen:
                        raise StartUpException(f"{f_name} must have a parameter named 'splash_screen' of type QtWidgets.QSplashScreen")

                    if typing.get_type_hints(f_value).get("return") != enum_state.StartUpState:
                        raise StartUpException(f"{f_name} must have a return type of StartUpState")

                    self.start_up_tasks.append(f_value)

    def showMessage(self, message, alignment=Qt.AlignmentFlag.AlignHCenter):
        super().showMessage(message, alignment)
        self.repaint()

    def startup(self, app: QApplication):
        self.show()
        self.showMessage("Loading...")

        logger.info("Start Up Registered Tasks", tasks=len(self.start_up_tasks))
        start = time.time()
        for task in self.start_up_tasks:
            logger.info("Start Up Registered Task", task=task.__name__)
            ret = task(self, app=app, assets=ASSETS_PATH)
            if ret is enum_state.StartUpState.SUCCESS:
                logger.info("Start Up Task Success", task=task.__name__)
                continue

            if ret is enum_state.StartUpState.FAILURE:
                logger.error("Start Up Task Failed", task=task.__name__)
                exit(0)
        time.sleep(1)
        self.showMessage("Start Up complete...")
        logger.info("Start Up Complete", time=time.time() - start)