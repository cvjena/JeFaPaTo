from pathlib import Path
from typing import Any, OrderedDict

import numpy as np
import pyqtgraph as pg
import structlog
from qtpy import QtWidgets

from jefapato import analyser, features, plotting

logger = structlog.get_logger()


class WidgetEyeBlinking(QtWidgets.QSplitter):
    def __init__(self):
        super().__init__()
        self.video_file_path: Path = None

        self.widget_frame = plotting.ImageBox()
        self.widget_face = plotting.ImageBox()
        self.widget_graph = plotting.WidgetGraph(add_yruler=False)

        self.plot_item = {}
        self.plot_data = {}

        self.vlayout_display = pg.GraphicsLayoutWidget()
        self.vlayout_display.addItem(self.widget_frame, row=0, col=0)
        self.vlayout_display.addItem(self.widget_face, row=0, col=1)
        self.vlayout_display.addItem(self.widget_graph, row=1, col=0, colspan=2)
        self.vlayout_display.ci.layout.setRowStretchFactor(0, 2)
        self.vlayout_display.ci.layout.setRowStretchFactor(1, 3)

        widget_r = QtWidgets.QWidget()

        self.vlayout_rs = QtWidgets.QVBoxLayout()
        self.flayout_se = QtWidgets.QFormLayout()

        widget_r.setLayout(self.vlayout_rs)

        self.addWidget(self.vlayout_display)
        self.addWidget(widget_r)

        self.cb_anal = QtWidgets.QCheckBox()
        self.bt_open = QtWidgets.QPushButton("Open Video File")
        self.bt_anal = QtWidgets.QPushButton("Analyze Video")
        self.bt_anal_stop = QtWidgets.QPushButton("Cancel")

        self.pb_anal = QtWidgets.QProgressBar()
        self.face_skipper = QtWidgets.QSpinBox()
        self.frame_skipper = QtWidgets.QSpinBox()

        self.vlayout_rs.addLayout(self.flayout_se)

        self.flayout_se.addRow(self.bt_open)
        self.flayout_se.addRow(self.bt_anal)
        self.flayout_se.addRow(self.bt_anal_stop)
        self.flayout_se.addRow("Skip Frame for Face Detection:", self.face_skipper)
        self.flayout_se.addRow("Skip Frames For Display:", self.frame_skipper)
        self.flayout_se.addRow("Progress:", self.pb_anal)

        self.features = [features.EARFeature]

        self.ea = analyser.LandmarkAnalyser(features=self.features)
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

    def setup_graph(self) -> None:
        self.widget_graph.clear()
        for k in self.plot_item:
            self.widget_graph.removeItem(k)

        self.plot_item.clear()
        self.plot_data.clear()

        for feature in self.features:
            for k, v in feature.plot_info.items():
                self.plot_item[k] = self.widget_graph.add_curve(**v)
                self.plot_data[k] = []

    @analyser.hookimpl
    def started(self):
        self.setup_graph()
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
    def updated(self, image: np.ndarray, face: np.ndarray):
        self.widget_frame.set_image(image)
        self.widget_face.set_image(face)

    @analyser.hookimpl
    def processed_percentage(self, percentage: int):
        self.pb_anal.setValue(percentage)

    @analyser.hookimpl
    def updated_feature(self, feature_data: OrderedDict[str, Any]) -> None:
        for feat in self.features:
            f_name = feat.__name__
            if f_name in feature_data:
                for k in feat.plot_info:
                    val = getattr(feature_data[f_name], k)
                    self.plot_data[k].append(val)
                    self.plot_item[k].setData(self.plot_data[k])

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
