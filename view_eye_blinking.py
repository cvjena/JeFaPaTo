import logging
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5 import uic
from PyQt5.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from jefapato.analyser import EyeBlinkingVideoAnalyser
from jefapato.plotter import EyeDetailWidget, FrameWidget, GraphWidget


class view_eye_blinking(QWidget):
    def __init__(self):
        super().__init__()

        # we set this for all images, as we do not expect
        # the images to be in a different orientation
        pg.setConfigOption(opt="imageAxisOrder", value="row-major")
        pg.setConfigOption(opt="background", value=pg.mkColor(255, 255, 255))

        # ==============================================================================
        # PROPERTIES
        self.current_image: np.ndarray = None
        self.video_file_path: Path = None

        self.disply_width = 640
        self.display_height = 480

        self.logger = logging.getLogger("eyeBlinkingDetection")
        # ==============================================================================
        # GUI ELEMENTS
        uic.loadUi("ui/view_eye_blinking.ui", self)

        self.widget_frame = FrameWidget()
        self.widget_detail = EyeDetailWidget()
        self.widget_graph = GraphWidget()

        # layouts
        self.hlayout_mw: QHBoxLayout = self.findChild(QHBoxLayout, "hlayout_mw")
        self.vlayout_left: QVBoxLayout = self.findChild(QVBoxLayout, "vlayout_left")
        self.vlayout_right: QVBoxLayout = self.findChild(QVBoxLayout, "vlayout_right")

        # edits
        self.edit_threshold: QLineEdit = self.findChild(QLineEdit, "edit_threshold")
        self.edit_bmp_l: QLineEdit = self.findChild(
            QLineEdit, "blinkingPerMinuteLeftLineEdit"
        )
        self.edit_bmp_r: QLineEdit = self.findChild(
            QLineEdit, "blinkingPerMinuteRightLineEdit"
        )

        # checkboxes
        self.checkbox_analysis: QCheckBox = self.findChild(
            QCheckBox, "checkbox_analysis"
        )

        # buttons
        self.button_video_load: QPushButton = self.findChild(
            QPushButton, "button_video_load"
        )
        self.button_video_analyze: QPushButton = self.findChild(
            QPushButton, "button_video_analyze"
        )

        self.button_video_analyze_stop: QPushButton = QPushButton("Stop Analyze")
        self.button_video_analyze_stop.setDisabled(True)
        self.button_video_analyze_stop.clicked.connect(self.stop_analyze)

        self.progressbar_analyze: QProgressBar = self.findChild(
            QProgressBar, "progressbar_analyze"
        )

        # ==============================================================================
        # Add widgets to layout
        self.vlayout_left.addWidget(self.widget_frame)
        self.vlayout_left.addWidget(self.widget_graph)
        self.vlayout_right.insertWidget(0, self.widget_detail)

        self.vlayout_right.insertWidget(4, self.button_video_analyze_stop)

        self.hlayout_mw.setStretchFactor(self.vlayout_left, 4)
        self.hlayout_mw.setStretchFactor(self.vlayout_right, 1)

        # ==============================================================================
        # INITIALIZATION ROUTINES
        # connect the functions
        # add slot connections

        self.ea = EyeBlinkingVideoAnalyser(
            self.widget_frame,
            self.widget_detail,
            self.widget_graph,
            self.edit_threshold,
        )

        self.ea.connect_on_started(
            [self.gui_analysis_start, self.progressbar_analyze.reset]
        )
        self.ea.connect_on_finished(
            [self.gui_analysis_finished, self.compute_blinking_per_minute]
        )
        self.ea.connect_processed_percentage([self.progressbar_analyze.setValue])

        self.button_video_load.clicked.connect(self.load_video)
        self.button_video_analyze.clicked.connect(self.start_anaysis)

        # disable analyse button and check box
        self.button_video_analyze.setDisabled(True)
        self.checkbox_analysis.setDisabled(True)

    def compute_blinking_per_minute(self):
        self.bpm_l, self.bpm_r = self.ea.blinking_rate()
        self.edit_bmp_l.setText(f"{self.bpm_l:5.2f}")
        self.edit_bmp_r.setText(f"{self.bpm_r:5.2f}")

    def start_anaysis(self):
        self.logger.info("User started analysis.")
        self.ea.analysis_start()

    def gui_analysis_start(self):
        self.button_video_load.setDisabled(True)
        self.button_video_analyze.setDisabled(True)
        self.button_video_analyze_stop.setDisabled(False)
        self.checkbox_analysis.setDisabled(True)
        self.edit_threshold.setDisabled(True)

    def gui_analysis_finished(self):
        self.button_video_load.setDisabled(False)
        self.button_video_analyze.setText("Video Analysieren")
        self.button_video_analyze.setDisabled(False)
        self.button_video_analyze_stop.setDisabled(True)

        self.checkbox_analysis.setDisabled(False)
        self.edit_threshold.setDisabled(False)

    def load_video(self):
        self.logger.info("Open file explorer")
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "Select video file",
            ".",
            "Video Files (*.mp4 *.flv *.ts *.mts *.avi *.mov)",
        )

        if fileName != "":
            self.logger.info(f"Load video file: {fileName}")
            self.video_file_path = Path(fileName)

            first_frame = self.ea.set_resource_path(self.video_file_path)
            self.button_video_analyze.setDisabled(False)
            self.checkbox_analysis.setDisabled(False)

            self.widget_frame.frame.set_image(first_frame, bgr=True)
        else:
            self.logger.info("No video file was selected")

    def stop_analyze(self) -> None:
        self.ea.stop()
