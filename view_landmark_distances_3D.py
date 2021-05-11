import os
import numpy as np
import dlib
import cv2
import shutil
import sys
import glob

from pathlib import Path

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5 import uic

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class view_landmark_distances_3D(QWidget):
    def __init__(self):
        super().__init__()

        self.landmark_file_path = ''
        self.image_list = []

        ## GUI ELEMENTS
        uic.loadUi("ui/view_landmark_distances_3D.ui", self)

        # label
        self.label_eye_left = self.findChild(QLabel, "label_eye_left")
        self.label_eye_right = self.findChild(QLabel, "label_eye_right")
        self.label_framenumber = self.findChild(QLabel, "label_framenumber")

        # edits
        self.edit_threshold = self.findChild(QLineEdit, "edit_threshhold")
        #self.edit_threshold.editingFinished.connect(self.change_threshold)

        # checkboxes
        self.checkbox_analysis: QCheckBox = self.findChild(QCheckBox, "checkbox_analysis")

        # buttons
        self.button_video_load = self.findChild(QPushButton, "button_video_load")
        self.button_video_analyze = self.findChild(QPushButton, "button_video_analyze")

        self.button_video_load.clicked.connect(self.load_folder)
        #self.button_video_analyze.clicked.connect(self.start_anaysis)

        # sliders
        self.slider_framenumber = self.findChild(QSlider, "slider_framenumber")
        #self.slider_framenumber.sliderMoved.connect(self.set_position)
        #self.slider_framenumber.sliderPressed.connect(self.set_position)

        # images
        self.view_video = self.findChild(QLabel, "view_video")
        self.view_face = self.findChild(QLabel, "view_face")
        self.view_eye_left = self.findChild(QLabel, "view_eye_left")
        self.view_eye_right = self.findChild(QLabel, "view_eye_right")

        # plotting
        self.evaluation_plot = MplCanvas(self, width=10, height=5, dpi=100)
        self.evaluation_plot.axes.plot([], [])

        self.vlayout_left = self.findChild(QVBoxLayout, "vlayout_left")
        self.vlayout_left.addWidget(self.evaluation_plot)

    def load_folder(self, ):
        print('load data from folder with 3D landmark file')
        landmark_file_path, _ = QFileDialog.getOpenFileName(self, "Select 3D landmark file",
                                                  ".", "3D landmark File (*.txt)")

        if landmark_file_path != '':
            self.landmark_file_path = Path(landmark_file_path)
            self.results_file_path = self.landmark_file_path.parent / (self.landmark_file_path.stem + ".csv")
            self.image_list = glob.glob(str(self.landmark_file_path.parent)+'*.png')



class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

        self.x_line: int = None
        self.y_line: int = None
        self.data_eye_left = []
        self.data_eye_right = []

    def set_eye_data(self, eye: str, data):
        if eye == "left":
            self.data_eye_left = data
        else:
            self.data_eye_right = data

    def set_xline(self, x_value):
        self.x_line = x_value

    def set_yline(self, y_value):
        self.y_line = y_value

    def plot(self):
        self.axes.clear()

        x_left = list(range(0, len(self.data_eye_left)))
        x_right = list(range(0, len(self.data_eye_right)))

        self.axes.plot(x_left, self.data_eye_left, c="blue")
        self.axes.plot(x_right, self.data_eye_right, c="red")

        if not self.x_line is None:
            self.axes.axvline(self.x_line)
        if not self.y_line is None:
            self.axes.axhline(self.y_line)

        self.axes.set_ylim(0, 6)

        self.draw()

    def clear(self):
        self.axes.clear()
