import pathlib
import sys
import time

import structlog
from qtpy import QtCore, QtGui, QtWidgets

import frontend
from frontend import start_up

logger = structlog.get_logger()


class JeFaPaTo(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JeFaPaTo - Jena Facial Palsy Tool")
        self.showMaximized()
        self.setMinimumSize(800, 600)

        self.central_widget = QtWidgets.QTabWidget()
        self.setCentralWidget(self.central_widget)

        # TODO: move to start_up and somehow make it easiert to be loaded
        self.VERSION = "2021.10.22"

        start = time.time()
        self.tab_eye_blinking = frontend.LandmarkExtraction()
        logger.info("Start Time LandmarkExtraction", time=time.time() - start)

        start = time.time()
        self.tab_eye_blinking_freq = frontend.WidgetEyeBlinkingFreq()
        logger.info("Start Time WidgetEyeBlinkingFreq", time=time.time() - start)

        # start = time.time()
        # self.tab_landmark_2d = frontend.WidgetLandmarkDistance2D()
        # logger.info("Start Time WidgetLandmarkDistance2D", time=time.time() - start)

        # start = time.time()
        # self.tab_landmark_3d = frontend.WidgetLandmarkDistance3D()
        # logger.info("Start Time WidgetLandmarkDistance3D", time=time.time() - start)

        # start = time.time()
        # self.tab_emotion_rec = frontend.WidgetEmotionRecognition()
        # logger.info("Start Time WidgetEmotionRecognition", time=time.time() - start)

        self.central_widget.addTab(self.tab_eye_blinking, "Landmark Extraction")
        self.central_widget.addTab(self.tab_eye_blinking_freq, "Eye Blinking Frequency")
        # self.addTab(self.tab_landmark_2d, "Landmark Analyses 2D")
        # self.addTab(self.tab_landmark_3d, "Landmark Analyses 3D")
        # self.addTab(self.tab_emotion_rec, "Emotion Recognition")

        self.central_widget.setCurrentIndex(0)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        logger.info("Close Event Detected", widget=self)
        logger.info("Shut Down Processes in each Tab")

        self.tab_eye_blinking.shut_down()
        self.tab_eye_blinking_freq.shut_down()
        # self.tab_landmark_2d.shut_down()
        # self.tab_landmark_3d.shut_down()
        # self.tab_emotion_rec.shut_down()

        logger.info("Internal Shut Down complete", widget=self)
        super().closeEvent(event)


class StartUpSplashScreen(QtWidgets.QSplashScreen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint
        )
        path = pathlib.Path(__file__).parent / "assets" / "splash.png"
        self.setPixmap(QtGui.QPixmap(path.as_posix()))
        self.setEnabled(False)

    def showMessage(self, message, alignment=QtCore.Qt.AlignHCenter):
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

        self.showMessage("Start Up complete...")
        logger.info("Start Up Complete", time=time.time() - start)


def main(argv):
    app = QtWidgets.QApplication(argv)

    splash = StartUpSplashScreen()
    splash.startup(app)

    ex = JeFaPaTo()

    logger.info("Start JeFaPaTo", version=ex.VERSION)

    ex.show()
    splash.finish(ex)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main(sys.argv)
