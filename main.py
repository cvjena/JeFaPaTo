import logging
import logging.config
import sys
from pathlib import Path

import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QTabWidget

from frontend.widget_emotion_recognition import WidgetEmotionRecognition
from frontend.widget_eye_blinking import WidgetEyeBlinking
from frontend.widget_eye_blinking_freq import WidgetEyeBlinkingFreq
from frontend.widget_landmark_distance_2d import WidgetLandmarkDistance2D
from frontend.widget_landmark_distance_3d import WidgetLandmarkDistance3D


class jefapato(QTabWidget):
    def __init__(self, parent=None):
        super(jefapato, self).__init__(parent)
        self.setWindowTitle("JeFaPaTo - Jena Facial Palsy Tool")
        self.showMaximized()

        # the images to be in a different orientation
        pg.setConfigOption(opt="imageAxisOrder", value="row-major")
        pg.setConfigOption(opt="background", value=pg.mkColor(255, 255, 255))

        self.VERSION = "2021.10.22"

        self.tab_eye_blinking = WidgetEyeBlinking()
        self.tab_eye_blinking_freq = WidgetEyeBlinkingFreq()
        self.tab_landmark_2d = WidgetLandmarkDistance2D()
        self.tab_landmark_3d = WidgetLandmarkDistance3D()
        self.tab_emotion_rec = WidgetEmotionRecognition()

        self.addTab(self.tab_eye_blinking, "Eye Blinking Extraction")
        self.addTab(self.tab_eye_blinking_freq, "Eye Blinking Frequency")
        self.addTab(self.tab_landmark_2d, "Landmark Analyses 2D")
        self.addTab(self.tab_landmark_3d, "Landmark Analyses 3D")
        self.addTab(self.tab_emotion_rec, "Emotion Recognition")

        self.setCurrentIndex(1)


def main(argv):
    log_path = Path("logs")
    log_path.mkdir(parents=True, exist_ok=True)

    logging.config.fileConfig(Path("config/logging.conf"))
    logger = logging.getLogger("root")

    app = QApplication(sys.argv)
    ex = jefapato()

    logger.info(f"Start JeFaPaTo version {ex.VERSION}")

    ex.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main(sys.argv)
