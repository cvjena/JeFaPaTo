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

        start = time.time()
        self.tab_eye_blinking = frontend.WidgetEyeBlinking()
        logger.info("Start Time WidgetEyeBlinking", time=time.time() - start)

        start = time.time()
        self.tab_eye_blinking_freq = frontend.WidgetEyeBlinkingFreq()
        logger.info("Start Time WidgetEyeBlinkingFreq", time=time.time() - start)

        start = time.time()
        self.tab_landmark_2d = frontend.WidgetLandmarkDistance2D()
        logger.info("Start Time WidgetLandmarkDistance2D", time=time.time() - start)

        start = time.time()
        self.tab_landmark_3d = frontend.WidgetLandmarkDistance3D()
        logger.info("Start Time WidgetLandmarkDistance3D", time=time.time() - start)

        start = time.time()
        self.tab_emotion_rec = frontend.WidgetEmotionRecognition()
        logger.info("Start Time WidgetEmotionRecognition", time=time.time() - start)

        self.addTab(self.tab_eye_blinking, "Eye Blinking Extraction")
        self.addTab(self.tab_eye_blinking_freq, "Eye Blinking Frequency")
        self.addTab(self.tab_landmark_2d, "Landmark Analyses 2D")
        self.addTab(self.tab_landmark_3d, "Landmark Analyses 3D")
        self.addTab(self.tab_emotion_rec, "Emotion Recognition")

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


def task_check_dlib_files(splash: StartUpSplashScreen):
    # check if dlib files are available
    # wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    # bunzip2 shape_predictor_68_face_landmarks.dat.bz2

    splash.showMessage("Checking dlib files...", alignment=QtCore.Qt.AlignHCenter)

    static_path = pathlib.Path(__file__).parent / "__static__"
    file_path = static_path / "shape_predictor_68_face_landmarks.dat"
    if file_path.exists() and file_path.is_file():
        splash.showMessage("Dlib files found", alignment=QtCore.Qt.AlignHCenter)
        time.sleep(0.5)
        return

    splash.showMessage("Download dlib files...", alignment=QtCore.Qt.AlignHCenter)
    logger.info("Downloading dlib shape predictor")
    # this is the case where we have to download the parameters!
    import bz2

    import requests

    static_path.mkdir(parents=True, exist_ok=True)

    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    res = requests.get(url)
    res.raise_for_status()
    logger.info("Downloaded dlib shape predictor")
    logger.info("Writing dlib shape predictor")

    splash.showMessage(
        "Writing dlib shape predictor...", alignment=QtCore.Qt.AlignHCenter
    )

    with open(file_path, "wb") as f:
        f.write(bz2.decompress(res.content))
    logger.info("Wrote dlib shape predictor", path=file_path)
    splash.showMessage("Dlib files written", alignment=QtCore.Qt.AlignHCenter)


def main(argv):
    app = QtWidgets.QApplication(argv)

    app.setWindowIcon(
        QtGui.QIcon(
            (pathlib.Path(__file__).parent / "assets" / "icon_256x256.png").as_posix()
        )
    )

    splash = StartUpSplashScreen()
    splash.show()

    splash.showMessage("Loading...", alignment=QtCore.Qt.AlignHCenter)
    time.sleep(0.2)

    task_check_dlib_files(splash)

    splash.showMessage("Starting...", alignment=QtCore.Qt.AlignHCenter)

    ex = jefapato()

    logger.info("Start JeFaPaTo", version=ex.VERSION)

    ex.show()
    splash.finish(ex)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main(sys.argv)
