import os
import numpy as np
# import dlib
import cv2
import shutil
import glob

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from eye_blinking_detector import EyeBlinkingDetector


class view_eye_blinking(QWidget):
    def __init__(self):
        super().__init__()

        # VIEW LAYOUT

        layout = QGridLayout()
        self.setLayout(layout)

        # ==============================================================================================================

        # PROBERTIES

        self.left_closed = False
        self.right_closed = False
        self.left_norm_distance = -1
        self.right_norm_distance = -1
        self.left_eye_closing_norm_area = -1
        self.right_eye_closing_norm_area = -1
        self.eye_distance_threshold_ratio = -1

        self.current_image = None
        self.video_file_path = None
        self.extract_folder = os.path.join('', 'tmp')
        self.image_paths = []
        self.video_fps = None

        self.results_file = None
        self.results_file_path = './results.csv'
        self.results_file_header = 'closed_left;closed_right;norm_eye_area_left;norm_eye_area_right\n'

        # ==============================================================================================================

        ## GUI ELEMENTS

        self.openButton = QPushButton("Open Video")
        self.openButton.setToolTip("Open Video File")
        self.openButton.setStatusTip("Open Video File")
        self.openButton.setFixedHeight(24)
        self.openButton.setIconSize(QSize(16, 16))
        self.openButton.setFont(QFont("Noto Sans", 8))
        self.openButton.setIcon(QIcon.fromTheme("document-open", QIcon("./")))
        self.openButton.clicked.connect(self.load_video)
        layout.addWidget(self.openButton, 0, 0)

        # video view
        self.disply_width = 640
        self.display_height = 480
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        # create a vertical box layout and add the two labels
        layout.addWidget(self.image_label, 1, 0)

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(2, 0)
        self.positionSlider.setToolTip(str(self.positionSlider.value()))
        self.positionSlider.sliderMoved.connect(self.set_position)
        self.positionSlider.sliderPressed.connect(self.set_position)
        layout.addWidget(self.positionSlider)

        self.label_slider_value = QLabel('0')
        layout.addWidget(self.label_slider_value, 3, 0)

        # PLOTTING
        self.plot_image_label = QLabel(self)
        self.plot_image_label.resize(self.disply_width, self.display_height)
        # create a vertical box layout and add the two labels
        layout.addWidget(self.plot_image_label, 4, 0)

        label_threshold = QLabel('Threshold:')
        layout.addWidget(label_threshold, 0, 1)
        self.edit_threshold = QLineEdit('8.75')
        self.edit_threshold.textChanged.connect(self.change_threshold)
        layout.addWidget(self.edit_threshold, 0, 2)

        self.startAnalysisButton = QPushButton("Start Analysis")
        self.startAnalysisButton.setToolTip("Start Analysis")
        self.startAnalysisButton.setStatusTip("Start Analysis")
        self.startAnalysisButton.setFixedHeight(24)
        self.startAnalysisButton.setIconSize(QSize(16, 16))
        self.startAnalysisButton.setFont(QFont("Noto Sans", 8))
        self.startAnalysisButton.setIcon(QIcon.fromTheme("document-open", QIcon("./")))
        self.startAnalysisButton.clicked.connect(self.start_anaysis)
        layout.addWidget(self.startAnalysisButton, 1, 1)

        label_left_eye_closed = QLabel('Left Eye:')
        layout.addWidget(label_left_eye_closed, 2, 1)
        self.edit_left_eye_closed = QLineEdit('open')
        layout.addWidget(self.edit_left_eye_closed, 2, 2)

        label_right_eye_closed = QLabel('Right Eye:')
        layout.addWidget(label_right_eye_closed, 3, 1)
        self.edit_right_eye_closed = QLineEdit('open')
        layout.addWidget(self.edit_right_eye_closed, 3, 2)

        # evaluation view
        self.evaluation_plot = MplCanvas(self, width=10, height=5, dpi=100)
        self.evaluation_plot.axes.plot([], [])
        layout.addWidget(self.evaluation_plot, 4, 0)
        # ==============================================================================================================

        ## INITIALIZATION ROUTINES

        self.eye_blinking_detector = EyeBlinkingDetector(float(self.edit_threshold.text()))

        if os.path.isfile(os.path.join(self.extract_folder, "frame_00000000.png")):
            self.show_image()

            # set slider
            self.positionSlider.setRange(0, len(self.image_paths) - 1)
            self.positionSlider.setSliderPosition(0)


    def start_anaysis(self):
        print('analyze all images ...')

        # open results output file and write header
        self.results_file = open(self.results_file_path, 'w')
        self.results_file.write(self.results_file_header)

        areas_left = []
        areas_right = []
        eye_distance_threshold_ratios = []

        # set the range of the slider
        # self.positionSlider.setRange(0, len(self.image_paths) - 1)

        for i_idx, image in enumerate(self.image_paths):
            print('image: ' + str(i_idx+1)+'/'+str(len(self.image_paths)))
            self.show_image(i_idx)
            self.positionSlider.setValue(i_idx)
            self.analyze_current_image()

            areas_left.append(self.left_eye_closing_norm_area)
            areas_right.append(self.right_eye_closing_norm_area)
            eye_distance_threshold_ratios.append(self.eye_distance_threshold_ratio)

            # write results to file
            self.results_file.write(self.edit_left_eye_closed.text() + ';'
                                   + self.edit_right_eye_closed.text() + ';'
                                   + str(self.left_eye_closing_norm_area) + ';'
                                   + str(self.right_eye_closing_norm_area)
                                   + '\n'
                                   )

        self.results_file.close()

        self.evaluation_plot.plot(list(range(0,len(areas_left))), areas_left)
        self.evaluation_plot.plot(list(range(0,len(areas_right))), areas_right)
        #self.evaluation_plot.plot(list(range(0,len(eye_distance_threshold_ratios))), eye_distance_threshold_ratios)
        #self.evaluation_plot.redraw()


    def calc_frequency(self):
        # calculates the frequency based on the number of eye closings and time
        frequency = 0

        return frequency

    def analyze_current_image(self):
        # print('analyze current image ...')
        self.left_closed, self.right_closed, self.left_eye_closing_norm_area, self.right_eye_closing_norm_area, self.eye_distance_threshold_ratio = self.eye_blinking_detector.detect_eye_blinking_in_image(
            self.current_image)
        if (self.left_closed):
            self.edit_left_eye_closed.setText("closed")
        else:
            self.edit_left_eye_closed.setText("open")

        if (self.right_closed):
            self.edit_right_eye_closed.setText("closed")
        else:
            self.edit_right_eye_closed.setText("open")

    def change_threshold(self):
        if not self.edit_threshold.text() == '':
            self.eye_blinking_detector.set_threshold(float(self.edit_threshold.text()))
            self.analyze_current_image()

    def set_position(self):
        self.show_image(self.positionSlider.value())
        self.analyze_current_image()
        plt.axvline(x=self.positionSlider.value())

    def load_video(self, ):
        print('load video from file')
        fileName, _ = QFileDialog.getOpenFileName(self, "Select video file",
                                                  ".", "Video Files (*.mp4 *.flv *.ts *.mts *.avi)")

        if fileName != '':
            self.video_file_path = fileName
            print('remove existing files ... ')
            existing_files = glob.glob(self.extract_folder+'/*')
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
        self.image_paths = glob.glob(os.path.join(self.extract_folder, '*.png'))
        self.image_paths.sort()

        self.label_slider_value.setText("Frame Number:\t" + str(image_id))

        cv_img = cv2.imread(self.image_paths[image_id])
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
        self.current_image = cv_img

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

    def plot(self, l1, l2):
        self.axes.plot(l1, l2)
        self.draw()


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

    def run(self):
        self.startAnalysisButton.setDisabled(True)

        vidcap = cv2.VideoCapture(self.video_file_path)
        success, image = vidcap.read()
        count = 0
        while success:
            self.openButton.setText(str(count))
            image_path = os.path.join(self.extract_folder, "frame_%08d.png" % count)
            self.image_paths.append(image_path)
            cv2.imwrite(image_path, image)
            success, image = vidcap.read()
            # print('Read a new frame: ', success)
            count += 1
        self.openButton.setText('Open Video')
        self.image_paths.sort()
        self.startAnalysisButton.setDisabled(False)
        # set the range of the slider
        self.positionSlider.setRange(0, len(self.image_paths) - 1)
