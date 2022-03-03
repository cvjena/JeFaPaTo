import pathlib
import sys
import time

import structlog
from qtpy import QtCore, QtGui, QtWidgets

import frontend
from frontend import start_up

logger = structlog.get_logger()


class jefapato(QtWidgets.QTabWidget):
    def __init__(self, parent=None):
        super(jefapato, self).__init__(parent)
        self.setWindowTitle("JeFaPaTo - Jena Facial Palsy Tool")
        self.showMaximized()
        self.setMinimumSize(800, 600)

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

        self.addTab(self.tab_eye_blinking, "Eye Blinking Extraction")
        self.addTab(self.tab_eye_blinking_freq, "Eye Blinking Frequency")
        # self.addTab(self.tab_landmark_2d, "Landmark Analyses 2D")
        # self.addTab(self.tab_landmark_3d, "Landmark Analyses 3D")
        # self.addTab(self.tab_emotion_rec, "Emotion Recognition")

        self.setCurrentIndex(0)


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


def main(argv):
    app = QtWidgets.QApplication(argv)

    splash = StartUpSplashScreen()
    splash.startup(app)

    ex = jefapato()

    logger.info("Start JeFaPaTo", version=ex.VERSION)

    ex.show()
    splash.finish(ex)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main(sys.argv)
