import argparse
import pathlib
import sys
import time

import git
import structlog
from qtpy import QtGui, QtWidgets
from PyQt6.QtCore import Qt


import frontend
from frontend import start_up

logger = structlog.get_logger()


# TODO move this to a separate file
class JeFaPaTo(QtWidgets.QMainWindow):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.setWindowTitle("JeFaPaTo - Jena Facial Palsy Tool")
        self.showMaximized()
        self.setMinimumSize(800, 600)

        self.central_widget = QtWidgets.QTabWidget()
        self.setCentralWidget(self.central_widget)

        self.progress_bar = QtWidgets.QProgressBar()

        self.VERSION = str(git.Repo(search_parent_directories=True).head.object.hexsha[:7]) # type: ignore

        start = time.time()
        self.tab_eye_blinking = frontend.LandmarkExtraction(self)
        logger.info("Start Time LandmarkExtraction", time=time.time() - start)

        start = time.time()
        self.tab_eye_blinking_freq = frontend.WidgetEyeBlinkingFreq(self)
        logger.info("Start Time WidgetEyeBlinkingFreq", time=time.time() - start)

        self.central_widget.addTab(self.tab_eye_blinking, "Landmark Extraction")
        self.central_widget.addTab(self.tab_eye_blinking_freq, "Eye Blinking Frequency")

        tab_idx = args.start_tab
        if tab_idx > self.central_widget.count():
            tab_idx = 0
        self.central_widget.setCurrentIndex(tab_idx)

        self.statusBar().addPermanentWidget(self.progress_bar)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        logger.info("Close Event Detected", widget=self)
        logger.info("Shut Down Processes in each Tab")

        self.tab_eye_blinking.shut_down()
        self.tab_eye_blinking_freq.shut_down()
        logger.info("Internal Shut Down complete", widget=self)
        super().closeEvent(event)


class StartUpSplashScreen(QtWidgets.QSplashScreen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
        path = pathlib.Path(__file__).parent / "assets" / "icons" / "icon_color.svg"
        self.setPixmap(QtGui.QPixmap(path.as_posix()))
        self.setEnabled(False)

    def showMessage(self, message, alignment=Qt.AlignmentFlag.AlignHCenter):
        super().showMessage(message, alignment)
        self.repaint()

    def startup(self, app: QtWidgets.QApplication):
        self.show()
        self.showMessage("Loading...")

        tasks = start_up.start_up_tasks
        logger.info("Start Up Registered Tasks", tasks=len(tasks))
        start = time.time()
        for task in tasks:
            logger.info("Start Up Registered Task", task=task.__name__)
            ret = task(self, app=app)
            if ret is start_up.StartUpState.SUCCESS:
                logger.info("Start Up Task Success", task=task.__name__)
                continue

            if ret is start_up.StartUpState.FAILURE:
                logger.error("Start Up Task Failed", task=task.__name__)
                exit(0)
        time.sleep(1)
        self.showMessage("Start Up complete...")
        logger.info("Start Up Complete", time=time.time() - start)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-tab", type=int, default=0)
    args = parser.parse_args(argv[1:])

    # TODO this is temporary and check if it can be handled somewhere else
    conf = pathlib.Path(__file__).parent / "__static__"
    conf.mkdir(parents=True, exist_ok=True)
    conf_f = conf / "conf.json"
    if not conf_f.exists():
        conf_f.touch(exist_ok=True)
        conf_f.write_text("{}")
    app = QtWidgets.QApplication(argv)
    app.setApplicationDisplayName("JeFaPaTo")
    app.setApplicationName("JeFaPaTo")
    splash = StartUpSplashScreen()
    splash.startup(app)

    ex = JeFaPaTo(args)

    logger.info("Start JeFaPaTo", version=ex.VERSION)

    ex.show()
    splash.finish(ex)
    sys.exit(app.exec_()) # type: ignore


if __name__ == "__main__":
    main(sys.argv)
