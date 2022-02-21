import pathlib
import sys
import time

import pyqtgraph as pg
import structlog
from qtpy import QtCore, QtGui, QtWidgets

import frontend

logger = structlog.get_logger()


class jefapato(QtWidgets.QTabWidget):
    def __init__(self, parent=None):
        super(jefapato, self).__init__(parent)
        self.setWindowTitle("JeFaPaTo - Jena Facial Palsy Tool")
        self.showMaximized()

        # the images to be in a different orientation
        pg.setConfigOption(opt="imageAxisOrder", value="row-major")
        pg.setConfigOption(opt="background", value=pg.mkColor(255, 255, 255))

        self.VERSION = "2021.10.22"

        self.tab_eye_blinking = frontend.WidgetEyeBlinking()
        self.tab_eye_blinking_freq = frontend.WidgetEyeBlinkingFreq()
        self.tab_landmark_2d = frontend.WidgetLandmarkDistance2D()
        self.tab_landmark_3d = frontend.WidgetLandmarkDistance3D()
        self.tab_emotion_rec = frontend.WidgetEmotionRecognition()

        self.addTab(self.tab_eye_blinking, "Eye Blinking Extraction")
        self.addTab(self.tab_eye_blinking_freq, "Eye Blinking Frequency")
        self.addTab(self.tab_landmark_2d, "Landmark Analyses 2D")
        self.addTab(self.tab_landmark_3d, "Landmark Analyses 3D")
        self.addTab(self.tab_emotion_rec, "Emotion Recognition")

        self.setCurrentIndex(1)


class StartUpSplashScreen(QtWidgets.QSplashScreen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint
        )
        path = pathlib.Path(__file__).parent / "assets" / "splash.png"
        self.setPixmap(QtGui.QPixmap(path.as_posix()))
        self.setEnabled(False)


def main(argv):
    app = QtWidgets.QApplication(argv)

    splash = StartUpSplashScreen()
    splash.show()

    splash.showMessage("Loading...", alignment=QtCore.Qt.AlignHCenter)

    for i in range(1, 11):
        # progressBar.setValue(i)
        t = time.time()
        while time.time() < t + 0.1:
            app.processEvents()
        splash.showMessage(f"Loading... {i*10:02d}%", alignment=QtCore.Qt.AlignHCenter)
    ex = jefapato()

    logger.info("Start JeFaPaTo", version=ex.VERSION)

    ex.show()
    splash.finish(ex)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main(sys.argv)
