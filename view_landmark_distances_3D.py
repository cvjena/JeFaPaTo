import os
import numpy as np
import cv2
import logging

from pathlib import Path


from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5 import uic

import pyqtgraph as pg
from pyqtgraph.GraphicsScene import mouseEvents

from landmark_distances import LandmarkDistance3D

class view_landmark_distances_3D(QWidget):
    def __init__(self):
        super().__init__()

        # ==============================================================================================================
        ## PROPERTIES
        self.current_image: np.ndarray = None
        self.landmark_file_path: Path = None
        self.image_files = []
        self.image_file_paths = []
        self.current_frame_idx = -1

        self.results_file = None
        self.results_file_path: Path = Path("./results_distances_3D.csv")
        self.results_file_header = 'left distance; right distance\n'

        self.display_width = 640
        self.display_height = 480

        self.landmark_distance_3D = LandmarkDistance3D()
        self.analyzer = Analyzer(self, self.landmark_distance_3D)

        self.logger = logging.getLogger("landmarkDistances3D")

        # ==============================================================================================================
        ## GUI ELEMENTS
        uic.loadUi("ui/view_landmark_distances_3D.ui", self)

        # label
        self.label_left: QLabel = self.findChild(QLabel, "label_left")
        self.label_right: QLabel = self.findChild(QLabel, "label_right")
        self.label_lm_idx_1: QLabel = self.findChild(QLabel, "label_lm_idx_1")
        self.label_lm_idx_2: QLabel = self.findChild(QLabel, "label_lm_idx_2")
        self.label_dist_left: QLabel = self.findChild(QLabel, "label_dist_left")
        self.label_dist_right: QLabel = self.findChild(QLabel, "label_dist_right")

        self.label_framenumber: QLabel = self.findChild(QLabel, "label_framenumber")

        # edits
        self.lineEdit_idx1_left: QLineEdit = self.findChild(QLineEdit, "lineEdit_idx1_left")
        self.lineEdit_idx1_right: QLineEdit = self.findChild(QLineEdit, "lineEdit_idx1_right")
        self.lineEdit_idx2_left: QLineEdit = self.findChild(QLineEdit, "lineEdit_idx2_left")
        self.lineEdit_idx2_right: QLineEdit = self.findChild(QLineEdit, "lineEdit_idx2_right")

        self.edit_threshold: QLineEdit = self.findChild(QLineEdit, "edit_threshhold")

        # checkboxes
        self.checkbox_analysis: QCheckBox = self.findChild(QCheckBox, "checkbox_analysis")

        # buttons
        self.button_lm_load: QPushButton = self.findChild(QPushButton, "button_lm_load")
        self.button_lm_analyze: QPushButton = self.findChild(QPushButton, "button_lm_analyze")

        # images
        self.view_image: QLabel = self.findChild(QLabel, "view_image")

        self.evaluation_plot = pg.PlotWidget()
        self.curve_left: pg.PlotDataItem = self.evaluation_plot.plot()
        self.curve_right: pg.PlotDataItem = self.evaluation_plot.plot()

        self.curve_left.setPen(pg.mkPen(pg.mkColor(0, 0, 255)))
        self.curve_right.setPen(pg.mkPen(pg.mkColor(255, 0, 0)))
        self.vertical_line: pg.InfiniteLine = pg.InfiniteLine(self.analyzer.frame, movable=True)
        self.horizontal_line: pg.InfiniteLine = pg.InfiniteLine(self.analyzer.threshold, angle=0, movable=True)
        self.evaluation_plot.addItem(self.vertical_line)
        self.evaluation_plot.addItem(self.horizontal_line)

        self.vertical_line.sigDragged.connect(self.change_frame)
        #self.horizontal_line.sigDragged.connect(self.change_threshold_per_line)

        self.vlayout_left: QVBoxLayout = self.findChild(QVBoxLayout, "vlayout_left")
        self.vlayout_left.addWidget(self.evaluation_plot)

        # ==============================================================================================================
        ## INITIALIZATION ROUTINES
        self.landmark_distance_3D.set_landmark_ids(int(self.lineEdit_idx1_left.text()),
                                                   int(self.lineEdit_idx2_left.text()),
                                                   int(self.lineEdit_idx1_right.text()),
                                                   int(self.lineEdit_idx2_right.text()))

        self.edit_threshold.editingFinished.connect(self.change_threshold_per_edit)

        self.button_lm_analyze.setDisabled(True)
        self.checkbox_analysis.setDisabled(True)

        # connect the functions
        self.button_lm_load.clicked.connect(self.load_landmark_file)
        self.button_lm_analyze.clicked.connect(self.start_anaysis)

    def change_frame(self):
        self.current_frame_idx = int(self.vertical_line.getPos()[0])
        self.update_distances()
        self.update_image()

    def update_image(self):
        if len(self.image_files) > self.current_frame_idx:
            image_path = os.path.join(str(self.landmark_file_path.parent), self.image_files[self.current_frame_idx])
            image = cv2.imread(image_path)
            img_frame: QPixmap     = self.convert_cv_qt(image, self.disply_width, self.display_height)
            self.view_image.setPixmap(img_frame)

    def update_distances(self):
        distance_left = self.landmark_distance_3D.get_distances_from_frame(self.current_frame_idx)[0]
        distance_right = self.landmark_distance_3D.get_distances_from_frame(self.current_frame_idx)[1]

        self.label_dist_left.setText(f"{distance_left:8.4f}".strip())
        self.label_dist_right.setText(f"{distance_right:8.4f}".strip())

    def load_landmark_file(self):
        self.logger.info("Open file explorer")
        fileName, _ = QFileDialog.getOpenFileName(self, "Select 3D landmark file",
                                                  ".", "Text Files (*.txt)")

        if fileName != '':
            self.logger.info(f"Load landmark file: {fileName}")
            self.landmark_file_path = Path(fileName)
            self.results_file_path = self.landmark_file_path.parent / (self.landmark_file_path.stem + ".csv")
            self.results_file_header = 'left distance (' + str(self.lineEdit_idx1_left.text()) + ',' + str(self.lineEdit_idx2_left.text()) + ');right distance ('  + str(self.lineEdit_idx1_right.text()) + ',' + str(self.lineEdit_idx2_right.text()) + ')\n'

            self.analyzer.reset()
            self.analyzer.set_landmark_file_path(self.landmark_file_path)
            self.analyzer.read_landmarks(self.landmark_file_path)

            # read image files if available
            image_files = [f for f in os.listdir(self.landmark_file_path.parent) if f.endswith('.png')]
            self.image_files = image_files.sort()

            self.button_lm_analyze.setDisabled(False)
            self.checkbox_analysis.setDisabled(False)
        else:
            self.logger.info(f"No landmark file was selected")

    def update_plot(self):
        self.curve_left.setData(self.analyzer.distance_left)
        self.curve_right.setData(self.analyzer.distance_right)
        self.vertical_line.setPos(self.analyzer.frame)
        self.horizontal_line.setPos(self.analyzer.threshold)

    def change_threshold_per_edit(self):
        try:
            self.analyzer.threshold = float(self.edit_threshold.text())
        except ValueError:
            self.edit_threshold.setText("Ungültige Zahl")
            return
        self.change_threshold()

    def change_threshold(self):
        self.landmark_distance_3D.set_threshold(self.analyzer.threshold)
        self.edit_threshold.setText(f"{self.analyzer.threshold:8.2f}".strip())
        if self.analyzer.has_run():
            self.button_video_analyze.setText("Erneut Analysieren")
        self.update_plot()

    def start_anaysis(self):
        self.thread_analyze = AnalyzeThread(self)
        self.thread_analyze.start()

class Analyzer():
    def __init__(self, veb: view_landmark_distances_3D, landmark_distance_3D: LandmarkDistance3D) -> None:
        self.veb = veb
        self.landmark_distance_3D = landmark_distance_3D

        self.distance_left = 0
        self.distance_right = 0
        self.distances_left = []
        self.distances_right = []
        self.current_frame = []
        self.landmark_coordinates = []

        self.landmark_distance_threshold_ratios = []

        self.landmark_file_path = None
        self.landmark_file = None
        self.frames_per_second = -1
        self.frames_total = -1

        self.threshold: float = 2.0
        self.frame: int = 0

        self.run_once = False

    def reset(self):
        self.run_once = False
        self.distances_left = []
        self.distances_right = []
        self.landmark_distance_threshold_ratios = []

    def analyze(self):
        self.landmark_distance_3D.get_distances_from_frame(self.current_frame)

    def analyze_complete(self):
        self.distance_left, self.distance_right = self.landmark_distance_3D.get_distances()

    def append_values(self):
        self.distances_left.append(self.distance_left)
        self.distances_right.append(self.distance_right)

    def set_landmark_file_path(self, path: Path):
        self.landmark_file_path = path

    def read_landmarks(self, path: Path):
        self.landmark_file_path = path
        self.landmark_coordinates = self.landmark_distance_3D.read_landmark_file(self.landmark_file_path)
        self.set_frame_total(len(self.landmark_coordinates))



    def set_frame_total(self, value):
        self.veb.logger.info(f"Landmark file contains {int(value)} frames")
        self.frames_total = int(value)

    def set_run(self):
        self.run_once = True

    def has_run(self):
        return self.run_once

    def save_results(self):
        # open results output file and write header
        self.results_file = open(self.veb.results_file_path, 'w')
        self.results_file.write(self.veb.results_file_header)
        for i in range(len(self.distance_left)):
            # fancy String literal concatenation
            line = (
                f"{self.distance_left[i]};"
                f"{self.distance_right[i]}"
                f"\n"
            )
            self.results_file.write(line)
        self.results_file.close()

class AnalyzeThread(QThread):
    def __init__(self, veb: view_landmark_distances_3D) -> None:
        super().__init__()
        self.veb = veb
        self.analyzer: Analyzer = self.veb.analyzer
        self.landmark_distance_3D: LandmarkDistance3D = self.veb.landmark_distance_3D

    def __del__(self):
        self.wait()

    def run(self):
        self.veb.logger.info(f"Analyse complete landmark file")
        # reset the values inside the analyzer
        self.analyzer.reset()
        self.analyzer.analyze_complete()

        self.veb.update_plot()

        self.analyzer.set_run()
        self.analyzer.save_results()
        #self.veb.button_lm_analyze.setText("Video Analysieren")
