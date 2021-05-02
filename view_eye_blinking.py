import os
import numpy as np
import dlib
import cv2
import shutil
import sys

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

        # create the tmp folder if it does not exists
        self.extract_folder = Path("tmp")
        self.extract_folder.mkdir(parents=True, exist_ok=True)

        self.image_paths = []
        self.video_fps = None

        self.results_file = None
        self.results_file_path: Path = Path("results.csv")
        self.results_file_header = 'closed_left;closed_right;norm_eye_area_left;norm_eye_area_right\n'

        self.disply_width = 640
        self.display_height = 480

        # ==============================================================================================================

        ## GUI ELEMENTS
        uic.loadUi("ui/view_eye_blinking.ui", self)

        # label
        self.label_eye_left  = self.findChild(QLabel, "label_eye_left")
        self.label_eye_right = self.findChild(QLabel, "label_eye_right")
        self.label_framenumber = self.findChild(QLabel, "label_framenumber")

        # edits
        self.edit_threshold = self.findChild(QLineEdit, "edit_threshhold")
        self.edit_threshold.editingFinished.connect(self.change_threshold)

        # checkboxes
        self.checkbox_analysis: QCheckBox= self.findChild(QCheckBox, "checkbox_analysis")
        
        # buttons
        self.button_video_load = self.findChild(QPushButton, "button_video_load")
        self.button_video_analyze = self.findChild(QPushButton, "button_video_analyze")

        self.button_video_load.clicked.connect(self.load_video)
        self.button_video_analyze.clicked.connect(self.start_anaysis)

        # sliders
        self.slider_framenumber = self.findChild(QSlider, "slider_framenumber")
        self.slider_framenumber.sliderMoved.connect(self.set_position)
        self.slider_framenumber.sliderPressed.connect(self.set_position)

        # images
        self.view_video = self.findChild(QLabel, "view_video")
        self.view_face  = self.findChild(QLabel, "view_face")
        self.view_eye_left  = self.findChild(QLabel, "view_eye_left")
        self.view_eye_right = self.findChild(QLabel, "view_eye_right")
        
        # plotting
        self.evaluation_plot = MplCanvas(self, width=10, height=5, dpi=100)
        self.evaluation_plot.axes.plot([], [])

        self.vlayout_left = self.findChild(QVBoxLayout, "vlayout_left")
        self.vlayout_left.addWidget(self.evaluation_plot)

        # ==============================================================================================================

        ## INITIALIZATION ROUTINES

        self.eye_blinking_detector = EyeBlinkingDetector(float(self.edit_threshold.text()))
        self.analyzer = Analyzer(self, self.eye_blinking_detector)

        # load the default value for the threshhold
        # check is not necessary as we have set the value in the 
        # UI file
        # FIXME set the default values in extra config file rather than UI file
        self.evaluation_plot.set_yline(float(self.edit_threshold.text()))
        self.evaluation_plot.plot()

        if (self.extract_folder / "frame_00000000.png").is_file():
            self.show_image()

            # set slider
            self.slider_framenumber.setRange(0, len(self.image_paths) - 1)
            self.slider_framenumber.setSliderPosition(0)

    def start_anaysis(self):
        self.thread_analyze = AnalyzeImagesThread(self)
        self.thread_analyze.start()
    
    def update_eye_labels(self):
        self.label_eye_left.setText(self.eye_blinking_detector.get_eye_left())
        self.label_eye_right.setText(self.eye_blinking_detector.get_eye_right())

    def update_plot(self):
        self.evaluation_plot.set_eye_data("left", self.analyzer.areas_left)
        self.evaluation_plot.set_eye_data("right", self.analyzer.areas_right)
        self.evaluation_plot.plot()

    def calc_frequency(self):
        # calculates the frequency based on the number of eye closings and time
        frequency = 0

        return frequency

    def change_threshold(self):
        try:
            input_value = float(self.edit_threshold.text())
            self.eye_blinking_detector.set_threshold(input_value)
            self.button_video_analyze.setText("Erneut Analysieren")
            self.evaluation_plot.set_yline(input_value)
            self.evaluation_plot.plot()
        except ValueError:
            self.edit_threshold.setText("Ungültige Zahl")

    def set_position(self):
        self.show_image(self.slider_framenumber.value())
        self.analyzer.analyze(self.current_image)
        self.update_eye_labels()

        self.evaluation_plot.set_xline(self.slider_framenumber.value())
        self.evaluation_plot.plot()

    def load_video(self, ):
        print('load video from file')
        fileName, _ = QFileDialog.getOpenFileName(self, "Select video file",
                                                  ".", "Video Files (*.mp4 *.flv *.ts *.mts *.avi)")

        if fileName != '':
            self.video_file_path = Path(fileName)
            self.results_file_path = self.video_file_path.parent / (self.video_file_path.stem + ".csv")

            self.analyzer.reset()
            print('remove existing files ... ')
            existing_files = self.extract_folder.glob('*')
            for f in existing_files:
                os.remove(f)
            self.extract_video()

    @pyqtSlot(list)
    def extract_video(self):
        if os.path.isdir(self.extract_folder):
            shutil.rmtree(self.extract_folder, ignore_errors=True)
            os.mkdir(self.extract_folder)

        self.extract_thread = ExtractImagesThread(self)
        self.extract_thread.image_paths_signal.connect(self.show_image)
        self.extract_thread.start()

    def show_image(self, image_id=0):

        # load the extracted frames as no new video has been loaded yet
        if len(self.image_paths) == 0:
            self.image_paths = sorted(self.extract_folder.glob('*.png'))

        self.label_framenumber.setText(f"Frame Number:\t {image_id:10d}")

        cv_img: np.ndarray = cv2.imread(self.image_paths[image_id].as_posix())
        qt_img = self.convert_cv_qt(cv_img, self.disply_width, self.display_height)
        self.view_video.setPixmap(qt_img)
        self.current_image = cv_img

        img_face: QPixmap      = self.convert_cv_qt(self.eye_blinking_detector.img_face, 200, 200)
        img_eye_left: QPixmap  = self.convert_cv_qt(self.eye_blinking_detector.img_eye_left, 100, 100)
        img_eye_right: QPixmap = self.convert_cv_qt(self.eye_blinking_detector.img_eye_right, 100, 100)

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

        self.frames_per_second = -1
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

    def analyze(self, image):
        self.detector.detect_eye_blinking_in_image(image)

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

    def set_run(self):
        self.run_once = True

    def has_run(self):
        return self.run_once

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
        if self.veb.checkbox_analysis.isChecked() or not self.analyzer.has_run():
            # reset the values inside the analyzer
            self.analyzer.reset()

            for i_idx, image in enumerate(self.veb.image_paths):
                #print('image: ' + str(i_idx+1)+'/'+str(len(self.veb.image_paths)))
                self.veb.show_image(i_idx)
                self.veb.slider_framenumber.setValue(i_idx)

                self.analyzer.analyze(self.veb.current_image)
                self.analyzer.append_values()

                self.veb.update_eye_labels()

                if i_idx % 5 == 0:
                    self.veb.update_plot()

            self.veb.update_plot()
            self.analyzer.set_run()

        else:
            self.analyzer.reset_closed()
            for i_idx, image in enumerate(self.veb.image_paths):
                self.veb.show_image(i_idx)
                self.veb.slider_framenumber.setValue(i_idx)
                self.analyzer.analyze_closing(i_idx)
                self.veb.update_eye_labels()

        self.analyzer.save_results()
        self.veb.button_video_analyze.setText("Video Analysieren")


class ExtractImagesThread(QThread):
    image_paths_signal = pyqtSignal(list)

    def __init__(self, veb: view_eye_blinking):
        super().__init__()
        self.veb = veb

    def run(self):
        self.image_paths = []
        self.veb.button_video_analyze.setDisabled(True)

        vidcap = cv2.VideoCapture(self.veb.video_file_path.as_posix())
        success, image = vidcap.read()

        self.veb.analyzer.set_frames_per_second(vidcap.get(cv2.CAP_PROP_FPS))

        count = 0
        while success:
            image_path = self.veb.extract_folder / f"frame_{count:08d}.png"
            cv2.imwrite(image_path.as_posix(), image)
            self.veb.button_video_load.setText(str(count))
            self.image_paths.append(image_path)
            success, image = vidcap.read()
            count += 1

        print(f"Loaded all {len(self.image_paths)} frames")
        self.image_paths.sort()

        self.veb.button_video_load.setText('Open Video')
        self.veb.button_video_analyze.setDisabled(False)
        self.veb.slider_framenumber.setRange(0, len(self.image_paths) - 1)
