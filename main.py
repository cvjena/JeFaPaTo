import sys

import logging
import logging.config

from pathlib import Path

import numpy as np

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from view_eye_blinking import view_eye_blinking
from view_landmark_distances_2D import view_landmark_distances_2D
from view_landmark_distances_3D import view_landmark_distances_3D
from view_emotion_recognition import view_emotion_recognition

class jefapato(QTabWidget):
    def __init__(self, parent=None):
        super(jefapato, self).__init__(parent)
        self.setWindowTitle("JeFaPaTo - Jena Facial Palsy Tool")

        self.VERSION = '2021.08.11'

        self.tab1 = view_eye_blinking()
        self.tab2 = view_landmark_distances_2D()
        self.tab3 = view_landmark_distances_3D()
        self.tab4 = view_emotion_recognition()

        self.addTab(self.tab1, "Eye Blinking Analyses")
        self.addTab(self.tab2, "Landmark Analyses 2D")
        self.addTab(self.tab3, "Landmark Analyses 3D")
        self.addTab(self.tab4, "Emotion Recognition")


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


if __name__ == '__main__':
    main(sys.argv)

