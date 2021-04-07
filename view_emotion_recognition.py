import numpy as np
import dlib

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


class view_emotion_recognition(QWidget):
    def __init__(self):
        super().__init__()

        layout = QHBoxLayout()
        layout.addWidget(QLabel("subjects"))
        layout.addWidget(QCheckBox("Physics"))
        layout.addWidget(QCheckBox("Maths"))

        self.setLayout(layout)

