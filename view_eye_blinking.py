from typing import Callable, List, Optional, Any, Tuple, Union, Type
import numpy as np
import cv2
import logging
from itertools import groupby

from pathlib import Path

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import * #type: ignore
from PyQt5.QtMultimedia import *
from PyQt5 import uic # type: ignore

import pyqtgraph as pg

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

        self.disply_width   = 640
        self.display_height = 480

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
        self.edit_threshold: QLineEdit = self.findChild(QLineEdit, "edit_threshold")
        self.edit_bmp_l: QLineEdit = self.findChild(QLineEdit, "blinkingPerMinuteLeftLineEdit")
        self.edit_bmp_r: QLineEdit = self.findChild(QLineEdit, "blinkingPerMinuteRightLineEdit")

        # checkboxes
        self.checkbox_analysis: QCheckBox = self.findChild(QCheckBox, "checkbox_analysis")
        
        # buttons
        self.button_video_load:    QPushButton = self.findChild(QPushButton, "button_video_load")
        self.button_video_analyze: QPushButton = self.findChild(QPushButton, "button_video_analyze")
        self.progressbar_analyze: QProgressBar = self.findChild(QProgressBar, "progressbar_analyze")
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
        self.evaluation_plot.setTitle("EAR Score")
        self.evaluation_plot.setMouseEnabled(x=True, y=False)
        self.evaluation_plot.setLimits(xMin=0)
        self.evaluation_plot.setYRange(0, 0.5)
        self.evaluation_plot.disableAutoRange()
        self.curve_left_eye:  pg.PlotDataItem = self.evaluation_plot.plot()
        self.curve_right_eye: pg.PlotDataItem = self.evaluation_plot.plot()
        self.curve_left_eye.setPen(pg.mkPen(pg.mkColor(0, 0, 255), width=2))
        self.curve_right_eye.setPen(pg.mkPen(pg.mkColor(255, 0, 0), width=2))

        self.evaluation_plot.scene().sigMouseClicked.connect(self.move_line)

        self.grid_item: pg.GridItem = pg.GridItem()
        self.grid_item.setTickSpacing(x=[1.0], y=[0.1])
        self.evaluation_plot.addItem(self.grid_item)

        bar_pen = pg.mkPen(width=2, color="k")
        self.indicator_frame: pg.InfiniteLine = pg.InfiniteLine(0, movable=False, pen=bar_pen)
        self.indicator_threshold: pg.InfiniteLine = pg.InfiniteLine(0.2, angle=0, movable=True, pen=bar_pen)

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
        self.evaluation_plot.addItem(self.indicator_frame)
        self.evaluation_plot.addItem(self.indicator_threshold)

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

        plotting = EyeBlinkingPlotting(
            plot=self.evaluation_plot,
            curve_eye_left=self.curve_left_eye,
            curve_eye_right=self.curve_right_eye,
            label_eye_left=self.label_eye_left,
            label_eye_right=self.label_eye_right,
            image_frame=self.image_frame,
            image_face=self.image_face,
            image_eye_left=self.image_eye_left,
            image_eye_right=self.image_eye_right,
            indicator_frame=self.indicator_frame,
            grid=self.grid_item
        )

        self.ea = EyeBlinkingVideoAnalyser(plotting)
        self.ea.connect_on_started([self.gui_analysis_start, self.progressbar_analyze.reset])
        self.ea.connect_on_finished([self.gui_analysis_finished, self.compute_blinking_per_minute])
        self.ea.connect_processed_percentage([self.progressbar_analyze.setValue])
        
        self.indicator_frame.sigDragged.connect(self.display_certain_frame)
        self.indicator_threshold.sigDragged.connect(self.change_threshold_per_line)

        self.button_video_load.clicked.connect(self.load_video)
        self.button_video_analyze.clicked.connect(self.start_anaysis)

        self.edit_threshold.editingFinished.connect(self.change_threshold_per_edit)

        # disable analyse button and check box
        self.button_video_analyze.setDisabled(True)
        self.checkbox_analysis.setDisabled(True)

    def compute_blinking_per_minute(self):
        frames_per_minute = int(self.ea.get_fps()) * 60
        amount_minutes = self.ea.get_data_amount() / frames_per_minute

        eye_l = [i[0] for i in groupby(self.ea.closed_eye_left)].count(True)
        eye_r = [i[0] for i in groupby(self.ea.closed_eye_right)].count(True)

        self.bpm_l = eye_l / amount_minutes
        self.bpm_r = eye_r / amount_minutes
        self.edit_bmp_l.setText(f"{self.bpm_l:5.2f}")
        self.edit_bmp_r.setText(f"{self.bpm_r:5.2f}")

    def move_line(self, mouseClickEvent):
        # this code  calculates the index of the underlying data entry
        # and moves the indicator to it
        vb = self.evaluation_plot.getPlotItem().vb
        mousePoint = vb.mapSceneToView(mouseClickEvent._scenePos)
        if self.evaluation_plot.sceneBoundingRect().contains(mouseClickEvent._scenePos):
            mousePoint = vb.mapSceneToView(mouseClickEvent._scenePos)
            index = int(mousePoint.x())
            self.indicator_frame.setPos(index)
            self.display_certain_frame()

    def display_certain_frame(self):
        self.ea.set_current_frame()

    def start_anaysis(self):
        self.ea.analysis_start()

    def gui_analysis_start(self):
        self.button_video_load.setDisabled(True)
        self.button_video_analyze.setDisabled(True)
        self.checkbox_analysis.setDisabled(True)
        self.edit_threshold.setDisabled(True)

        self.indicator_frame.setMovable(False)
        self.indicator_threshold.setMovable(False)

    def gui_analysis_finished(self):
        self.button_video_load.setDisabled(False)
        self.button_video_analyze.setText("Video Analysieren")
        self.button_video_analyze.setDisabled(False)
        self.checkbox_analysis.setDisabled(False)
        self.edit_threshold.setDisabled(False)

        self.indicator_frame.setMovable(True)
        self.indicator_threshold.setMovable(True)

    def change_threshold_per_edit(self):
        try:
            value: float = float(self.edit_threshold.text())
            self.ea.set_threshold(value)
            self.indicator_threshold.setPos(value)
        except ValueError:
            self.edit_threshold.setText("Ungültige Zahl")
            return
        self.change_threshold()

    def change_threshold_per_line(self):
        self.ea.set_threshold(float(self.indicator_threshold.getPos()[1]))
        self.change_threshold()

    def change_threshold(self):
        self.edit_threshold.setText(f"{self.ea.get_threshold():8.2f}".strip())

    def load_video(self):
        self.logger.info("Open file explorer")
        fileName, _ = QFileDialog.getOpenFileName(self, "Select video file",
                                                  ".", "Video Files (*.mp4 *.flv *.ts *.mts *.avi *.mov)")

        if fileName != '':
            self.logger.info(f"Load video file: {fileName}")
            self.video_file_path = Path(fileName)
            
            self.ea.set_resource_path(self.video_file_path)
            self.button_video_analyze.setDisabled(False)
            self.checkbox_analysis.setDisabled(False)
        else:
            self.logger.info(f"No video file was selected")

from jefapato.analyser import VideoAnalyser
from jefapato.feature_extractor import LandMarkFeatureExtractor, scale_bbox
from jefapato.classifier import EyeBlinkingResult, EyeBlinking68Classifier

import dlib

from dataclasses import dataclass
@dataclass
class EyeBlinkingPlotting:
    plot: pg.PlotWidget
    curve_eye_left:  pg.PlotDataItem
    curve_eye_right: pg.PlotDataItem

    label_eye_left:  pg.LabelItem
    label_eye_right: pg.LabelItem

    image_frame:     pg.ImageItem
    image_face:      pg.ImageItem
    image_eye_left:  pg.ImageItem
    image_eye_right: pg.ImageItem

    indicator_frame: pg.InfiniteLine

    grid: pg.GridItem

class EyeBlinkingVideoAnalyser(VideoAnalyser):
    def __init__(self, plotting: EyeBlinkingPlotting) -> None:
        
        self.eye_blinking_classifier = EyeBlinking68Classifier(threshold=0.2)
        super().__init__(
            feature_extractor=LandMarkFeatureExtractor(), 
            classifier=self.eye_blinking_classifier
            )

        self.results_file_header = 'closed_left;closed_right;norm_eye_area_left;norm_eye_area_right\n'

        self.plotting: EyeBlinkingPlotting = plotting

        self.connect_on_started([self.__on_start])
        self.connect_on_updated([self.__on_update])
        self.connect_on_finished([self.__on_finished])

        self.score_eye_left:  List[float] = list()
        self.score_eye_right: List[float] = list()

        self.closed_eye_left:  List[bool] = list()
        self.closed_eye_right: List[bool] = list()

        self.current_frame: int = 0

    def set_threshold(self, value:float):
        self.eye_blinking_classifier.threshold = value
    
    def get_threshold(self) -> float:
        return self.eye_blinking_classifier.threshold

    def __on_start(self):
        self.score_eye_left  = list()
        self.score_eye_right = list()
        self.closed_eye_left = list()
        self.closed_eye_right = list()

        self.plotting.plot.enableAutoRange(axis="x")
        self.plotting.plot.setMouseEnabled(x=False, y=False)
        self.plotting.grid.setTickSpacing(
            x=[self.get_fps()],
            y=None
        )

    def __on_update(self, frame: np.ndarray, frame_id: int):
        # currently we only check the first one
        # TODO make possible for more faces
        self.current_frame = frame_id
        latest_value: EyeBlinkingResult = self.classes[-1][0]
        
        self.score_eye_left.append(latest_value.ear_left)
        self.score_eye_right.append(latest_value.ear_right)

        self.closed_eye_left.append(latest_value.closed_left)
        self.closed_eye_right.append(latest_value.closed_right)

        # plotting of the data
        # TODO move to extra functions for cleaner structure and easier reuse
        self.plotting.indicator_frame.setPos(self.current_frame)

        self.plotting.curve_eye_left.setData(self.score_eye_left)
        self.plotting.curve_eye_right.setData(self.score_eye_right)

        self.plot_label()

        # TODO plotting of the frame and extracted faces
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rect, shape = self.features[-1][0]

        self.plot_frame(frame, rect, shape)
        self.plotting.plot.setLimits(xMin=self.current_frame-100, xMax=self.current_frame)
       
    def set_current_frame(self) -> None:
        # clip the frame into the maximum valid range
        # cast to int, just to be sure
        value = int(max(0, min(int(self.plotting.indicator_frame.pos()[0]), self.data_amount-1)))
        self.plotting.indicator_frame.setPos(value) 
        self.current_frame = value
        if not self.resource_is_loaded():
            return

        self.set_next_item_by_id(self.current_frame)
        (grabbed, frame) = self.get_next_item()

        # TODO do something better if nothing gets grabbed
        if not grabbed:
            return
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            rect, shape = self.features[self.current_frame][0]
        except IndexError as e:
            logging.exception("User probably clicked on the plot without anaylizing the video. Silent error.")
            return

        self.plot_label()
        self.plot_frame(frame, rect, shape)

    def plot_label(self):
        self.plotting.label_eye_left.setText(self.closed_text(self.closed_eye_left[self.current_frame]))
        self.plotting.label_eye_right.setText(self.closed_text(self.closed_eye_right[self.current_frame]))

    def plot_frame(self, frame: np.ndarray, rect: dlib.rectangle, shape: np.ndarray) -> None:
        face = np.copy(frame)

        eye_left  = shape[self.eye_blinking_classifier.eye_left_slice]
        eye_right = shape[self.eye_blinking_classifier.eye_right_slice]
        # get the outer region of the plot
        eye_left_mean  = np.nanmean(eye_left, axis=0).astype(np.int32)
        eye_right_mean = np.nanmean(eye_right, axis=0).astype(np.int32)

        self.draw_shape(face, eye_left,  color=(0, 0, 255))
        self.draw_shape(face, eye_right, color=(255, 0, 0))

        eye_left_width  = (np.nanmax(eye_left, axis=0)[0] - np.nanmin(eye_left, axis=0)[0]) // 2
        eye_right_width = (np.nanmax(eye_right, axis=0)[0] - np.nanmin(eye_right, axis=0)[0]) // 2

        bbox_eye_left = scale_bbox(
            bbox=dlib.rectangle(eye_left_mean[0] - eye_left_width, eye_left_mean[1] - eye_left_width, eye_left_mean[0] + eye_left_width, eye_left_mean[1] + eye_left_width,),
            scale=1,
            padding=-1
        )
        bbox_eye_right = scale_bbox(
            bbox=dlib.rectangle(eye_right_mean[0] - eye_right_width, eye_right_mean[1] - eye_right_width, eye_right_mean[0] + eye_right_width, eye_right_mean[1] + eye_right_width,),
            scale=1,
            padding=-1
        )
        img_eye_left  = face[bbox_eye_left.top():bbox_eye_left.bottom(), bbox_eye_left.left():bbox_eye_left.right()]
        img_eye_right = face[bbox_eye_right.top():bbox_eye_right.bottom(), bbox_eye_right.left():bbox_eye_right.right()]
        face = face[rect.top():rect.bottom(), rect.left():rect.right()]

        self.plotting.image_frame.setImage(frame)
        self.plotting.image_face.setImage(face)
        self.plotting.image_eye_left.setImage(img_eye_left)
        self.plotting.image_eye_right.setImage(img_eye_right)

    def closed_text(self, value: bool) -> str:
        return "closed" if value else "open"

    def __on_finished(self):
        self.save_results()
        self.plotting.plot.setLimits(xMin=0, xMax=self.current_frame)
        self.plotting.plot.setXRange(self.current_frame-100, self.current_frame)
        self.plotting.plot.setMouseEnabled(x=True, y=False)

    def save_results(self):
        resource_path = self.get_resource_path()
        result_path = resource_path.parent / (resource_path.stem + ".csv")

        with open(result_path, 'w') as f:
            f.write(self.results_file_header)
            for i in range(len(self.score_eye_right)):
                # fancy String literal concatenation
                line = (
                        f"{'closed' if self.closed_eye_left[i] else 'open'};"
                        f"{'closed' if self.closed_eye_right[i] else 'open'};"
                        f"{self.score_eye_left[i]};"
                        f"{self.score_eye_right[i]}"
                        f"\n"
                    )
                f.write(line)

    def draw_shape(self, img: np.ndarray, shape: np.ndarray, color: Tuple) -> None:
        for (x, y) in shape:
            cv2.circle(img, (x, y), 1, color, -1)