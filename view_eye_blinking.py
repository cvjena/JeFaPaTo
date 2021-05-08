
import numpy as np
import cv2
import logging

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

from eye_blinking_detector import EyeBlinkingDetector


class view_eye_blinking(QWidget):
    def __init__(self):
        super().__init__()
        # ==============================================================================================================
        ## PROPERTIES
        self.current_image: np.ndarray = None
        self.video_file_path: Path = None

        self.results_file = None
        self.results_file_path: Path = Path("results.csv")
        self.results_file_header = 'closed_left;closed_right;norm_eye_area_left;norm_eye_area_right\n'

        self.disply_width = 640
        self.display_height = 480

        self.logger = logging.getLogger("eyeBlinkingDetection")
        # ==============================================================================================================
        ## GUI ELEMENTS
        uic.loadUi("ui/view_eye_blinking.ui", self)

        # label
        self.label_eye_left:    QLabel = self.findChild(QLabel, "label_eye_left")
        self.label_eye_right:   QLabel = self.findChild(QLabel, "label_eye_right")
        self.label_framenumber: QLabel = self.findChild(QLabel, "label_framenumber")

        # edits
        self.edit_threshold: QLineEdit = self.findChild(QLineEdit, "edit_threshhold")

        # checkboxes
        self.checkbox_analysis: QCheckBox = self.findChild(QCheckBox, "checkbox_analysis")
        
        # buttons
        self.button_video_load:    QPushButton = self.findChild(QPushButton, "button_video_load")
        self.button_video_analyze: QPushButton = self.findChild(QPushButton, "button_video_analyze")

        # sliders
        self.slider_framenumber: QSlider = self.findChild(QSlider, "slider_framenumber")

        # images
        self.view_video:     QLabel = self.findChild(QLabel, "view_video")
        self.view_face:      QLabel = self.findChild(QLabel, "view_face")
        self.view_eye_left:  QLabel = self.findChild(QLabel, "view_eye_left")
        self.view_eye_right: QLabel = self.findChild(QLabel, "view_eye_right")
        
        # plotting
        self.evaluation_plot = MplCanvas(self, width=10, height=5, dpi=100)
        self.evaluation_plot.axes.plot([], [])

        self.vlayout_left: QVBoxLayout = self.findChild(QVBoxLayout, "vlayout_left")
        self.vlayout_left.addWidget(self.evaluation_plot)

        # ==============================================================================================================
        ## INITIALIZATION ROUTINES
        self.eye_blinking_detector = EyeBlinkingDetector(float(self.edit_threshold.text()))
        self.analyzer = Analyzer(self, self.eye_blinking_detector)

        # connect the functions
        self.button_video_load.clicked.connect(self.load_video)
        self.button_video_analyze.clicked.connect(self.start_anaysis)

        self.edit_threshold.editingFinished.connect(self.change_threshold)
        
        self.slider_framenumber.sliderMoved.connect(self.set_position)
        self.slider_framenumber.sliderPressed.connect(self.set_position)

        # disable analyse button and check box
        self.button_video_analyze.setDisabled(True)
        self.checkbox_analysis.setDisabled(True)

        # load the default value for the threshhold
        # check is not necessary as we have set the value in the 
        # UI file
        # FIXME set the default values in extra config file rather than UI file
        self.evaluation_plot.set_yline(float(self.edit_threshold.text()))
        self.evaluation_plot.plot()

        self.show_image()

    def start_anaysis(self):
        self.thread_analyze = AnalyzeImagesThread(self)
        self.thread_analyze.start()
    
    def update_eye_labels(self):
        self.label_eye_left.setText(self.eye_blinking_detector.get_eye_left())
        self.label_eye_right.setText(self.eye_blinking_detector.get_eye_right())

    def update_plot(self):
        self.evaluation_plot.set_eye_data("left", self.analyzer.areas_left)
        self.evaluation_plot.set_eye_data("right", self.analyzer.areas_right)
        self.evaluation_plot.set_xline(self.slider_framenumber.value())
        self.evaluation_plot.plot()

    def change_threshold(self):
        try:
            input_value = float(self.edit_threshold.text())
            self.eye_blinking_detector.set_threshold(input_value)
            if self.analyzer.has_run():
                self.button_video_analyze.setText("Erneut Analysieren")
            self.evaluation_plot.set_yline(input_value)
            self.evaluation_plot.plot()
        except ValueError:
            self.edit_threshold.setText("Ungültige Zahl")

    def set_position(self):
        # load the new frame by the given slider id
        self.analyzer.set_frame_by_id(self.slider_framenumber.value())
        self.analyzer.analyze()

        self.update_eye_labels()
        self.update_plot()

    def load_video(self):
        self.logger.info("Open file explorer")
        fileName, _ = QFileDialog.getOpenFileName(self, "Select video file",
                                                  ".", "Video Files (*.mp4 *.flv *.ts *.mts *.avi)")

        if fileName != '':
            self.logger.info(f"Load video file: {fileName}")
            self.video_file_path = Path(fileName)
            self.results_file_path = self.video_file_path.parent / (self.video_file_path.stem + ".csv")

            self.analyzer.reset()
            self.analyzer.set_video(self.video_file_path)
            self.slider_framenumber.setRange(0, self.analyzer.frames_total - 1)

            self.button_video_analyze.setDisabled(False)
            self.checkbox_analysis.setDisabled(False)
        else:
            self.logger.info(f"No video file was selected")

    def show_image(self):
        img_frame: QPixmap     = self.convert_cv_qt(self.analyzer.current_frame, self.disply_width, self.display_height)
        img_face: QPixmap      = self.convert_cv_qt(self.analyzer.current_face, 200, 200)
        img_eye_left: QPixmap  = self.convert_cv_qt(self.analyzer.current_eye_left, 100, 100)
        img_eye_right: QPixmap = self.convert_cv_qt(self.analyzer.current_eye_right, 100, 100)

        self.view_video.setPixmap(img_frame)
        self.view_face.setPixmap(img_face)
        self.view_eye_left.setPixmap(img_eye_left)
        self.view_eye_right.setPixmap(img_eye_right)


    def convert_cv_qt(self, cv_img, width, height):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(width, height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


class Analyzer():
    def __init__(self, veb: view_eye_blinking, detector: EyeBlinkingDetector) -> None:
        self.veb = veb
        self.detector = detector
        
        self.left_closed = []
        self.right_closed = []

        self.areas_left = []
        self.areas_right = []
        self.eye_distance_threshold_ratios = []

        self.video = None
        self.frames_per_second = -1
        self.frames_total = -1

        self.current_frame: np.ndarray = np.zeros((480, 640, 3), dtype=np.uint8)
        self.current_face:  np.ndarray = np.zeros((100, 100, 3), dtype=np.uint8)
        self.current_eye_left:  np.ndarray = np.zeros((50, 50, 3), dtype=np.uint8)
        self.current_eye_right: np.ndarray = np.zeros((50, 50, 3), dtype=np.uint8)

        self.run_once = False

    def reset(self):
        self.run_once = False
        self.left_closed = []
        self.right_closed = []

        self.areas_left = []
        self.areas_right = []
        self.eye_distance_threshold_ratios = []

    def reset_closed(self):
        self.left_closed = []
        self.right_closed = []

    def analyze(self):
        self.detector.detect_eye_blinking_in_image(self.current_frame)
        self.current_face = self.detector.img_face
        self.current_eye_left  = self.detector.img_eye_left
        self.current_eye_right = self.detector.img_eye_right

    def analyze_closing(self, image_idx: int):
        l, r = self.detector.check_closing(
            region_left=self.areas_left[image_idx], 
            region_right=self.areas_right[image_idx],
            eye_distance=self.eye_distance_threshold_ratios[image_idx]
        )
        self.left_closed.append(l)
        self.right_closed.append(r)

    def append_values(self):
        self.left_closed.append(self.detector.left_closed)
        self.right_closed.append(self.detector.right_closed)

        self.areas_left.append(self.detector.left_eye_closing_norm_area)
        self.areas_right.append(self.detector.right_eye_closing_norm_area)
        self.eye_distance_threshold_ratios.append(self.detector.eye_distance_threshold_ratio)

    def set_frames_per_second(self, value):
        self.frames_per_second = value

    def set_frame_total(self, value):
        self.veb.logger.info(f"Video contains {int(value)} frames")
        self.frames_total = int(value)

    def set_video(self, path: Path):
        self.video = cv2.VideoCapture(path.as_posix())
        self.set_frames_per_second(self.video.get(cv2.CAP_PROP_FPS))
        self.set_frame_total(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_current_frame(self) -> np.ndarray:
        return self.current_frame

    def set_frame_by_id(self, id: int):
        # TODO better return if the frame is out of range
        if not (0 <= id < self.frames_total):
            self.veb.logger.warning(f"Frame {id} not in range of {0} to {self.frames_total}")
            self.current_frame = np.zeros((100,100, 3), dtype=np.uint8)

        # set the current frame we want to extract for the video file
        # https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html#aa6480e6972ef4c00d74814ec841a2939
        # https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
        self.video.set(cv2.CAP_PROP_POS_FRAMES, id)
        success, frame = self.video.read()

        # TODO same as above
        if not success:
            self.veb.logger.error(f"Frame {id} could not be loaded")
            self.current_frame = np.zeros((100,100, 3), dtype=np.uint8)

        self.current_frame = frame

    def set_run(self):
        self.run_once = True

    def has_run(self):
        return self.run_once

    def calc_frequency(self):
        # calculates the frequency based on the number of eye closings and time
        frequency = 0

        return frequency

    def save_results(self):
        # open results output file and write header
        self.results_file = open(self.veb.results_file_path, 'w')
        self.results_file.write(self.veb.results_file_header)
        for i in range(len(self.left_closed)):
            # fancy String literal concatenation
            line = (
                    f"{'closed' if self.left_closed[i] else 'open'};"
                    f"{'closed' if self.left_closed[i] else 'open'};"
                    f"{self.areas_left[i]};"
                    f"{self.areas_right[i]}"
                    f"\n"
                )
            self.results_file.write(line)
        self.results_file.close()


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

        self.x_line: int = None
        self.y_line: int = None
        self.data_eye_left = []
        self.data_eye_right = []

    def set_eye_data(self, eye:str, data):
        if eye=="left":
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

class AnalyzeImagesThread(QThread):
    def __init__(self, veb: view_eye_blinking) -> None:
        super().__init__()
        self.veb = veb
        self.analyzer: Analyzer = self.veb.analyzer
        self.detector: EyeBlinkingDetector = self.veb.eye_blinking_detector

    def __del__(self):
        self.wait()

    def run(self):
        # Disable the button so the user cannto click it again
        # TODO move this code rather to the gui and call it with a function!
        # self.veb.button_video_analyze.setDisabled(True)
        # self.veb.checkbox_analysis.setDisabled(True)
        # self.veb.edit_threshold.setDisabled(True)
        if self.veb.checkbox_analysis.isChecked() or not self.analyzer.has_run():
            self.veb.logger.info(f"Analyse complete video")
            # reset the values inside the analyzer
            self.analyzer.reset()

            for i_idx in range(self.analyzer.frames_total):
                self.analyzer.set_frame_by_id(i_idx)
                self.analyzer.analyze()
                self.analyzer.append_values()

                self.veb.slider_framenumber.setValue(i_idx)
                self.veb.show_image()
                self.veb.update_eye_labels()

                # FIXME Updating the plot takes most of the time during this calculation...
                if i_idx % 5 == 0:
                    self.veb.update_plot()

            #self.veb.update_plot()
            self.analyzer.set_run()

        else:
            self.veb.logger.info(f"Re-analyse eye closing")
            self.analyzer.reset_closed()
            for i_idx in range(self.analyzer.frames_total):
                self.analyzer.analyze_closing(i_idx)
                self.analyzer.set_frame_by_id(i_idx)
                self.veb.slider_framenumber.setValue(i_idx)

                self.veb.show_image()
                self.veb.update_eye_labels()
                #self.veb.update_plot()

        self.analyzer.save_results()
        self.veb.button_video_analyze.setText("Video Analysieren")
        # Re-enable the analyse button for the user
        # self.veb.button_video_analyze.setDisabled(False)
        # self.veb.checkbox_analysis.setDisabled(False)
        # self.veb.edit_threshold.setDisabled(False)