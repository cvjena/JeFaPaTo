import numpy as np
#import dlib

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


class view_landmark_distances(QWidget):
    def __init__(self):
        super().__init__()

        layout = QFormLayout()
        #sex = QHBoxLayout()
        #sex.addWidget(QRadioButton("Male"))
        #sex.addWidget(QRadioButton("Female"))
        #layout.addRow(QLabel("Sex"), sex)
        #layout.addRow("Date of Birth", QLineEdit())
        #self.setTabText(1, "Personal Details")
        #self.tab2.setLayout(layout)
        self.setLayout(layout)

