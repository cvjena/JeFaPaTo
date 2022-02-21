from pathlib import Path
from typing import Any, OrderedDict

import numpy as np
import structlog
from qtpy import QtWidgets

from jefapato import analyser, features, plotter

logger = structlog.get_logger()


class WidgetEyeBlinking(QtWidgets.QSplitter):
    def __init__(self):
        super().__init__()
        self.video_file_path: Path = None

        self.widget_frame = plotter.FrameWidget()
        self.widget_detail = plotter.EyeDetailWidget()
        self.widget_graph = plotter.GraphWidget(add_yruler=False)

        widget_l = QtWidgets.QWidget()
        widget_r = QtWidgets.QWidget()

        self.vlayout_ls = QtWidgets.QVBoxLayout()
        self.vlayout_rs = QtWidgets.QVBoxLayout()
        self.flayout_se = QtWidgets.QFormLayout()

        widget_l.setLayout(self.vlayout_ls)
        widget_r.setLayout(self.vlayout_rs)

        self.addWidget(widget_l)
        self.addWidget(widget_r)

        self.cb_anal = QtWidgets.QCheckBox()
        self.bt_open = QtWidgets.QPushButton("Open Video File")
        self.bt_anal = QtWidgets.QPushButton("Analyze Video")
        self.bt_anal_stop = QtWidgets.QPushButton("Cancel")

        self.pb_anal = QtWidgets.QProgressBar()
        self.face_skipper = QtWidgets.QSpinBox()
        self.frame_skipper = QtWidgets.QSpinBox()

        self.vlayout_ls.addWidget(self.widget_frame)
        self.vlayout_ls.addWidget(self.widget_graph)

        self.vlayout_rs.addWidget(self.widget_detail)
        self.vlayout_rs.addLayout(self.flayout_se)

        self.flayout_se.addRow(self.bt_open)
        self.flayout_se.addRow(self.bt_anal)
        self.flayout_se.addRow(self.bt_anal_stop)
        self.flayout_se.addRow("Skip Frame for Face Detection:", self.face_skipper)
        self.flayout_se.addRow("Skip Frames For Display:", self.frame_skipper)
        self.flayout_se.addRow("Progress:", self.pb_anal)

        self.ea = analyser.LandmarkAnalyser(features=[features.EARFeature])
        self.ea.register_hooks(self)

        self.bt_open.clicked.connect(self.load_video)
        self.bt_anal.clicked.connect(self.ea.start)
        self.bt_anal_stop.clicked.connect(self.ea.stop)

        self.face_skipper.setRange(3, 20)
        self.face_skipper.setValue(5)
        self.face_skipper.valueChanged.connect(self.ea.set_face_detect_skip)

        self.frame_skipper.setRange(1, 20)
        self.frame_skipper.setValue(5)
        self.frame_skipper.valueChanged.connect(self.ea.set_frame_skip)

        # disable analyse button and check box
        self.bt_anal.setDisabled(True)
        self.cb_anal.setDisabled(True)
        self.bt_anal_stop.setDisabled(True)

        self.setStretchFactor(0, 7)
        self.setStretchFactor(1, 3)

    @analyser.hookimpl
    def started(self):
        self.bt_open.setDisabled(True)
        self.bt_anal.setDisabled(True)
        self.bt_anal_stop.setDisabled(False)
        self.cb_anal.setDisabled(True)
        self.face_skipper.setDisabled(True)
        self.frame_skipper.setDisabled(True)

    @analyser.hookimpl
    def finished(self):
        self.bt_open.setDisabled(False)
        self.bt_anal.setText("Analyze Video")
        self.bt_anal.setDisabled(False)
        self.bt_anal_stop.setDisabled(True)

        self.cb_anal.setDisabled(False)
        self.face_skipper.setDisabled(False)
        self.frame_skipper.setDisabled(False)

    @analyser.hookimpl
    def updated(self, data: np.ndarray, features: np.ndarray):
        self.widget_frame.frame.set_image(data)

    @analyser.hookimpl
    def processed_percentage(self, percentage: int):
        self.pb_anal.setValue(percentage)

    @analyser.hookimpl
    def update_feature(self, features: OrderedDict[str, Any]) -> None:
        logger.info("Got the feature data", keys=features.keys())

    def load_video(self):
        logger.info("Open File Dialog", widget=self)
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select video file",
            ".",
            "Video Files (*.mp4 *.flv *.ts *.mts *.avi *.mov)",
        )

        if fileName != "":
            logger.info("Video file selected", file_name=fileName)
            self.video_file_path = Path(fileName)

            self.ea.set_resource_path(self.video_file_path)
            self.bt_anal.setDisabled(False)
            self.cb_anal.setDisabled(False)
        else:
            logger.info("Open File Dialog canceled")
