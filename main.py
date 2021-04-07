import sys
import os
import numpy as np
#import dlib
import cv2


from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from view_eye_blinking import view_eye_blinking
from view_landmark_distances import view_landmark_distances
from view_emotion_recognition import view_emotion_recognition

class tabdemo(QTabWidget):
    def __init__(self, parent=None):
        super(tabdemo, self).__init__(parent)
        self.setWindowTitle("JeFaPaTo - Jena Facial Palsy Tool")
        self.window().showMaximized()

        self.tab1 = view_eye_blinking(camera_id=0)
        self.tab2 = view_landmark_distances()
        self.tab3 = view_emotion_recognition()

        self.addTab(self.tab1, "Eye Blinking Analyses")
        self.addTab(self.tab2, "Landmark Analyses")
        self.addTab(self.tab3, "Emotion Recognition")


def main(argv):
    app = QApplication(sys.argv)
    ex = tabdemo()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main(sys.argv)

