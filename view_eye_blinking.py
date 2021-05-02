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
        self.analyzer = Analyzer(self.eye_blinking_detector)

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
        x_left = list(range(0, len(self.analyzer.areas_left)))
        x_right = list(range(0, len(self.analyzer.areas_right)))

        self.evaluation_plot.clear()

        self.evaluation_plot.plot(x_left, self.analyzer.areas_left, color="blue")
        self.evaluation_plot.plot(x_right, self.analyzer.areas_right, color="red")

    def calc_frequency(self):
        # calculates the frequency based on the number of eye closings and time
        frequency = 0

        return frequency

    def change_threshold(self):
        if not self.edit_threshold.text() == '':
            self.eye_blinking_detector.set_threshold(float(self.edit_threshold.text()))
            self.analyze_current_image()

    def set_position(self):
        self.show_image(self.slider_framenumber.value())
        self.analyzer.analyze(self.current_image)
        self.update_eye_labels()
        plt.axvline(x=self.slider_framenumber.value())

    def load_video(self, ):
        print('load video from file')
        fileName, _ = QFileDialog.getOpenFileName(self, "Select video file",
                                                  ".", "Video Files (*.mp4 *.flv *.ts *.mts *.avi)")

        if fileName != '':
            self.video_file_path = Path(fileName)
            self.results_file_path = self.video_file_path.parent / (self.video_file_path.stem + ".csv")

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

        face_img = self.convert_cv_qt(self.eye_blinking_detector.face_imge, 200, 200)
        self.view_face.setPixmap(face_img)


    def convert_cv_qt(self, cv_img, width, height):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(width, height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


class Analyzer():
    def __init__(self, detector: EyeBlinkingDetector) -> None:
        self.detector = detector
        
        self.areas_left = []
        self.areas_right = []
        self.eye_distance_threshold_ratios = []

    def reset(self):
        self.areas_left = []
        self.areas_right = []
        self.eye_distance_threshold_ratios = []

    def analyze(self, image):
        self.detector.detect_eye_blinking_in_image(image)

    def append_values(self):
        self.areas_left.append(self.detector.left_eye_closing_norm_area)
        self.areas_right.append(self.detector.right_eye_closing_norm_area)
        self.eye_distance_threshold_ratios.append(self.detector.eye_distance_threshold_ratio)


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

    def plot(self, l1, l2, color):
        self.axes.plot(l1, l2, c=color)
        self.draw()

    def clear(self):
        self.axes.clear()

class AnalyzeImagesThread(QThread):
    def __init__(self, veb: view_eye_blinking) -> None:
        super().__init__()
        self.veb = veb
        self.analyzer = self.veb.analyzer
        self.detector = self.veb.eye_blinking_detector

    def __del__(self):
        self.wait()

    def run(self):
        print('analyze all images ...')

        # reset the values inside the analyzer
        self.analyzer.reset()

        # open results output file and write header
        self.results_file = open(self.veb.results_file_path, 'w')
        self.results_file.write(self.veb.results_file_header)

        for i_idx, image in enumerate(self.veb.image_paths):
            print('image: ' + str(i_idx+1)+'/'+str(len(self.veb.image_paths)))
            self.veb.show_image(i_idx)
            self.veb.slider_framenumber.setValue(i_idx)
            
            self.analyzer.analyze(self.veb.current_image)
            self.analyzer.append_values()
            self.veb.update_eye_labels()

            # fancy String literal concatenation¶
            line = (
                f"{self.detector.get_eye_left};"
                f"{self.detector.get_eye_right};"
                f"{self.detector.left_eye_closing_norm_area};"
                f"{self.detector.right_eye_closing_norm_area}"
                f"\n"
            )
            self.veb.update_plot()
            # write results to file
            self.results_file.write(line)

        self.results_file.close()


class ExtractImagesThread(QThread):
    image_paths_signal = pyqtSignal(list)

    def __init__(self, view_eye_blinking):
        super().__init__()
        self.openButton = view_eye_blinking.openButton
        self.video_file_path = view_eye_blinking.video_file_path
        self.image_paths = view_eye_blinking.image_paths
        self.extract_folder = view_eye_blinking.extract_folder
        self.startAnalysisButton = view_eye_blinking.startAnalysisButton
        self.positionSlider = view_eye_blinking.positionSlider
        self.video_fps = -1

    def run(self):
        self.image_paths = []
        self.startAnalysisButton.setDisabled(True)

        vidcap = cv2.VideoCapture(self.video_file_path.as_posix())
        success, image = vidcap.read()

        self.video_fps = vidcap.get(cv2.CAP_PROP_FPS)
        print(f"Frames per second: {self.video_fps:5.2f}")

        count = 0
        while success:
            self.openButton.setText(str(count))
            image_path = self.extract_folder / f"frame_{count:08d}.png"
            self.image_paths.append(image_path)
            cv2.imwrite(image_path.as_posix(), image)
            success, image = vidcap.read()
            # print('Read a new frame: ', success)
            count += 1

        print("Loaded all frames")
        self.openButton.setText('Open Video')
        self.image_paths.sort()
        self.startAnalysisButton.setDisabled(False)
        # set the range of the slider
        self.positionSlider.setRange(0, len(self.image_paths) - 1)
