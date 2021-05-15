
from typing import List, Optional
import numpy as np
import cv2
import logging

import time
from queue import Queue
from threading import Thread

from pathlib import Path

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import * #type: ignore
from PyQt5.QtMultimedia import *
from PyQt5 import uic # type: ignore

import pyqtgraph as pg
from pyqtgraph.GraphicsScene import mouseEvents
from pyqtgraph.graphicsItems.ViewBox.ViewBox import ViewBox

from eye_blinking_detector import EyeBlinkingDetector


class view_eye_blinking(QWidget):
    def __init__(self):
        super().__init__()

        # we set this for all images, as we do not expect
        # the images to be in a different orientation
        pg.setConfigOption(
            opt="imageAxisOrder", value="row-major"
        )
        pg.setConfigOption(
            opt="background", value=pg.mkColor(255, 255, 255)
        )

        # ==============================================================================================================
        ## PROPERTIES
        self.current_image: np.ndarray = None
        self.video_file_path: Path = None

        self.results_file = None
        self.results_file_path: Path = Path("results_eye_blinking.csv")
        self.results_file_header = 'closed_left;closed_right;norm_eye_area_left;norm_eye_area_right\n'

        self.disply_width   = 640
        self.display_height = 480

        self.eye_blinking_detector = EyeBlinkingDetector()
        self.analyzer = Analyzer(self, self.eye_blinking_detector)

        self.logger = logging.getLogger("eyeBlinkingDetection")
        # ==============================================================================================================
        ## GUI ELEMENTS
        uic.loadUi("ui/view_eye_blinking.ui", self)

        # layouts
        self.hlayout_mw:    QHBoxLayout = self.findChild(QHBoxLayout, "hlayout_mw")
        self.vlayout_left:  QVBoxLayout = self.findChild(QVBoxLayout, "vlayout_left")
        self.vlayout_right: QVBoxLayout = self.findChild(QVBoxLayout, "vlayout_right")

        self.layout_frame:  pg.GraphicsLayoutWidget = pg.GraphicsLayoutWidget(title="Video Frame")
        self.layout_detail: pg.GraphicsLayoutWidget = pg.GraphicsLayoutWidget()
        self.layout_eye_left:  pg.GraphicsLayout = pg.GraphicsLayout()
        self.layout_eye_right: pg.GraphicsLayout = pg.GraphicsLayout()

        # label
        self.label_eye_left:  pg.LabelItem = pg.LabelItem("open")
        self.label_eye_right: pg.LabelItem = pg.LabelItem("open")

        # edits
        self.edit_threshold: QLineEdit = self.findChild(QLineEdit, "edit_threshhold")

        # checkboxes
        self.checkbox_analysis: QCheckBox = self.findChild(QCheckBox, "checkbox_analysis")
        
        # buttons
        self.button_video_load:    QPushButton = self.findChild(QPushButton, "button_video_load")
        self.button_video_analyze: QPushButton = self.findChild(QPushButton, "button_video_analyze")

        # image holders
        image_settings = {
            "invertY": True, 
            "lockAspect":True,
            "enableMouse": False,
        }

        self.view_frame: pg.ViewBox = pg.ViewBox(**image_settings)
        self.view_face:  pg.ViewBox = pg.ViewBox(**image_settings)
        self.view_image_left:  pg.ViewBox() = pg.ViewBox(**image_settings)
        self.view_image_right: pg.ViewBox() = pg.ViewBox(**image_settings)
        # images
        self.image_frame:     pg.ImageItem = pg.ImageItem()
        self.image_face:      pg.ImageItem = pg.ImageItem()
        self.image_eye_left:  pg.ImageItem = pg.ImageItem()
        self.image_eye_right: pg.ImageItem = pg.ImageItem()

        # plotting
        self.evaluation_plot: pg.PlotWidget   = pg.PlotWidget()
        self.curve_left_eye:  pg.PlotDataItem = self.evaluation_plot.plot()
        self.curve_right_eye: pg.PlotDataItem = self.evaluation_plot.plot()
        self.curve_left_eye.setPen(pg.mkPen(pg.mkColor(0, 0, 255), width=2))
        self.curve_right_eye.setPen(pg.mkPen(pg.mkColor(255, 0, 0), width=2))

        bar_pen = pg.mkPen(width=2, color="k")
        self.vertical_line:   pg.InfiniteLine = pg.InfiniteLine(self.analyzer.frame, movable=False, pen=bar_pen)
        self.horizontal_line: pg.InfiniteLine = pg.InfiniteLine(self.analyzer.threshold, angle=0, movable=True, pen=bar_pen)

        # ==============================================================================================================
        # Add widgets to layout
        self.vlayout_left.addWidget(self.layout_frame)
        self.vlayout_left.addWidget(self.evaluation_plot)
        self.vlayout_right.insertWidget(0, self.layout_detail)
        
        self.hlayout_mw.setStretchFactor(self.vlayout_left, 4)
        self.hlayout_mw.setStretchFactor(self.vlayout_right, 1)
        # view the frame
        self.layout_frame.addItem(self.view_frame)

        # add the sliders to the plot
        self.evaluation_plot.addItem(self.vertical_line)
        self.evaluation_plot.addItem(self.horizontal_line)

        # add the images to the image holders
        self.view_frame.addItem(self.image_frame)
        self.view_face.addItem(self.image_face)
        self.view_image_left.addItem(self.image_eye_left)
        self.view_image_right.addItem(self.image_eye_right)

        # setup the detail layout
        self.layout_detail.addItem(self.view_face, row=0, col=0, rowspan=2, colspan=2)
        self.layout_detail.addItem(self.layout_eye_left,  row=2, col=1)
        self.layout_detail.addItem(self.layout_eye_right, row=2, col=0)

        # detail right eye
        self.layout_eye_right.addLabel(text='Right eye',     row=0, col=0)
        self.layout_eye_right.addItem(self.view_image_right, row=1, col=0)
        self.layout_eye_right.addItem(self.label_eye_right,  row=2, col=0)

        # detail left eye
        self.layout_eye_left.addLabel(text='Left eye',     row=0, col=0)
        self.layout_eye_left.addItem(self.view_image_left, row=1, col=0)
        self.layout_eye_left.addItem(self.label_eye_left,  row=2, col=0)

        # ==============================================================================================================
        ## INITIALIZATION ROUTINES
        # connect the functions
        # add slot connections
        self.vertical_line.sigDragged.connect(self.change_frame)
        self.horizontal_line.sigDragged.connect(self.change_threshold_per_line)

        self.button_video_load.clicked.connect(self.load_video)
        self.button_video_analyze.clicked.connect(self.start_anaysis)

        self.edit_threshold.editingFinished.connect(self.change_threshold_per_edit)

        # disable analyse button and check box
        self.button_video_analyze.setDisabled(True)
        self.checkbox_analysis.setDisabled(True)

        self.show_image()

    def change_frame(self):
        self.analyzer.set_frame_by_id(id=int(self.vertical_line.getPos()[0]))
        self.analyzer.analyze()
        self.update_eye_labels()
        self.update_plot()
        self.show_image()

    def start_anaysis(self):

        thread_load = AnalyzeImageLoaderThread(self)
        thread_load.start()

        self.thread_analyze = AnalyzeImagesThread(self)
        self.thread_analyze.analysisUpdated.connect(self.update_eye_labels)
        self.thread_analyze.analysisUpdated.connect(self.update_plot)
        self.thread_analyze.analysisUpdated.connect(self.show_image)

        self.thread_analyze.analysisStarted.connect(self.gui_analysis_start)
        self.thread_analyze.analysisFinished.connect(self.gui_analysis_finished)

        self.thread_analyze.start()
    
    def gui_analysis_start(self):
        self.button_video_load.setDisabled(True)
        self.button_video_analyze.setDisabled(True)
        self.checkbox_analysis.setDisabled(True)
        self.edit_threshold.setDisabled(True)

        self.vertical_line.setMovable(False)
        self.horizontal_line.setMovable(False)

    def gui_analysis_finished(self):
        self.button_video_load.setDisabled(False)
        self.button_video_analyze.setText("Video Analysieren")
        self.button_video_analyze.setDisabled(False)
        self.checkbox_analysis.setDisabled(False)
        self.edit_threshold.setDisabled(False)

        self.vertical_line.setMovable(True)
        self.horizontal_line.setMovable(True)

    def update_eye_labels(self):
        self.label_eye_left.setText(self.eye_blinking_detector.get_eye_left())
        self.label_eye_right.setText(self.eye_blinking_detector.get_eye_right())

    def update_plot(self):
        self.curve_left_eye.setData(self.analyzer.areas_left)
        self.curve_right_eye.setData(self.analyzer.areas_right)
        self.vertical_line.setPos(self.analyzer.frame)
        self.horizontal_line.setPos(self.analyzer.threshold)

    def change_threshold_per_edit(self):
        try:
            self.analyzer.threshold = float(self.edit_threshold.text())
        except ValueError:
            self.edit_threshold.setText("Ungültige Zahl")    
            return
        self.change_threshold()

    def change_threshold_per_line(self):
        self.analyzer.threshold = float(self.horizontal_line.getPos()[1])
        self.change_threshold()

    def change_threshold(self):
        self.edit_threshold.setText(f"{self.analyzer.threshold:8.2f}".strip())
        if self.analyzer.has_run():
            self.button_video_analyze.setText("Erneut Analysieren")
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

            self.button_video_analyze.setDisabled(False)
            self.checkbox_analysis.setDisabled(False)
        else:
            self.logger.info(f"No video file was selected")

    def show_image(self):
        self.image_frame.setImage(cv2.cvtColor(self.analyzer.current_frame, cv2.COLOR_BGR2RGB))
        self.image_face.setImage(cv2.cvtColor(self.analyzer.current_face, cv2.COLOR_BGR2RGB))
        self.image_eye_left.setImage(cv2.cvtColor(self.analyzer.current_eye_left, cv2.COLOR_BGR2RGB))
        self.image_eye_right.setImage(cv2.cvtColor(self.analyzer.current_eye_right, cv2.COLOR_BGR2RGB))
        
class Analyzer():
    def __init__(self, veb: view_eye_blinking, detector: EyeBlinkingDetector) -> None:
        self.veb = veb
        self.detector = detector
        
        self.left_closed: List  = []
        self.right_closed: List = []

        self.areas_left: List  = []
        self.areas_right: List = []

        self.video: cv2.VideoCapture = None
        self.frames_per_second = -1
        self.frames_total = -1

        self.threshold: float = 0.2
        self.frame: int = 0

        self.frame_queue: Queue = Queue(maxsize=0)

        self.current_frame: np.ndarray = np.zeros((480, 640, 3), dtype=np.uint8)
        self.current_face:  np.ndarray = np.zeros((100, 100, 3), dtype=np.uint8)
        self.current_eye_left:  np.ndarray = np.zeros((50, 50, 3), dtype=np.uint8)
        self.current_eye_right: np.ndarray = np.zeros((50, 50, 3), dtype=np.uint8)

        self.run_once = False

    def reset(self, reset_only_closing: Optional[bool]=False):
        self.left_closed = []
        self.right_closed = []

        if reset_only_closing:
            return

        self.run_once = False
        self.areas_left = []
        self.areas_right = []

    def analyze(self):
        self.detector.detect_eye_blinking_in_image(self.current_frame, self.threshold)
        self.current_face = self.detector.img_face
        self.current_eye_left  = self.detector.img_eye_left
        self.current_eye_right = self.detector.img_eye_right

    def analyze_closing(self, image_idx: int):
        l, r = self.detector.check_closing(
            score_left=self.areas_left[image_idx], 
            score_right=self.areas_right[image_idx],
            threshold=self.threshold
        )
        self.left_closed.append(l)
        self.right_closed.append(r)

    def append_values(self):
        self.left_closed.append(self.detector.left_closed)
        self.right_closed.append(self.detector.right_closed)

        self.areas_left.append(self.detector.left_eye_closing_norm_area)
        self.areas_right.append(self.detector.right_eye_closing_norm_area)

    def set_frames_per_second(self, value):
        self.frames_per_second = value

    def set_frame_total(self, value):
        self.veb.logger.info(f"Video contains {int(value)} frames")
        self.frames_total = int(value)

    def set_video(self, path: Path):
        if self.video is not None:
            self.video.release()

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
            return
        self.frame = id
        # set the current frame we want to extract for the video file
        # https://docs.opencv.org/4.5.1/d8/dfe/classcv_1_1VideoCapture.html#aa6480e6972ef4c00d74814ec841a2939
        # https://docs.opencv.org/4.5.1/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
        if self.video is None:
            self.current_frame = np.zeros((100,100, 3), dtype=np.uint8)
            return 

        self.video.set(cv2.CAP_PROP_POS_FRAMES, id)
        success, frame = self.video.read()

        # TODO same as above
        if not success:
            self.veb.logger.error(f"Frame {id} could not be loaded")
            self.current_frame = np.zeros((100,100, 3), dtype=np.uint8)
            return 

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

class AnalyzeImageLoaderThread(Thread):
    def __init__(self, veb: view_eye_blinking) -> None:
        super().__init__()
        self.veb = veb
        self.analyzer: Analyzer = self.veb.analyzer
        self.stopped = False

    def run(self):
        self.analyzer.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            if self.stopped:
                return
            
            if not self.analyzer.frame_queue.full():
                (grabbed, frame) = self.analyzer.video.read()
                
                if not grabbed:
                    return

                self.analyzer.frame_queue.put(frame)

class AnalyzeImagesThread(QThread):
    analysisUpdated:  pyqtSignal = pyqtSignal()
    analysisStarted:  pyqtSignal = pyqtSignal()
    analysisFinished: pyqtSignal = pyqtSignal()

    def __init__(self, veb: view_eye_blinking) -> None:
        super().__init__()
        self.veb = veb
        self.analyzer: Analyzer = self.veb.analyzer
        self.detector: EyeBlinkingDetector = self.veb.eye_blinking_detector

    def __del__(self):
        self.wait()

    def run(self):
        self.analysisStarted.emit()
        self.veb.logger.info(f"Analyse complete video")
        # reset the values inside the analyzer

        complete_run: bool = self.veb.checkbox_analysis.isChecked() or not self.analyzer.has_run()
        self.analyzer.reset(reset_only_closing=(not complete_run))

        processed = 0

        while processed < self.analyzer.frames_total:
            if self.analyzer.frame_queue.qsize() > 0:
                self.analyzer.current_frame = self.analyzer.frame_queue.get_nowait()
                self.analyzer.frame_queue.task_done()
                if complete_run:
                    self.analyzer.analyze()
                    self.analyzer.append_values()
                else:
                    self.analyzer.analyze_closing(processed)
                self.analysisUpdated.emit()
                self.analyzer.frame = processed
                processed += 1

        self.analysisUpdated.emit()
        self.analyzer.set_run()

        self.analyzer.save_results()
        self.analysisFinished.emit()