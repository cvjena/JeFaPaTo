__all__ = ["LandmarkExtraction"]

import csv
import datetime
import pathlib
from typing import Any, List, OrderedDict, Type

import numpy as np
import pyqtgraph as pg
import qtawesome as qta
import structlog
from qtpy import QtCore, QtWidgets

from jefapato import analyser, config, features, plotting

logger = structlog.get_logger()


class FeatureCheckBox(QtWidgets.QCheckBox):
    def __init__(self, feature: Type[features.Feature], **kwargs):
        super().__init__(**kwargs)
        self.feature = feature
        self.setText(feature.__name__)


class LandmarkExtraction(QtWidgets.QSplitter, config.Config):
    updated = QtCore.Signal(int)

    def __init__(self):
        config.Config.__init__(self, prefix="landmarks")
        QtWidgets.QSplitter.__init__(self)

        self.video_file_path: pathlib.Path = None

        self.widget_frame = plotting.ImageBox()
        self.widget_face = plotting.ImageBox()
        self.widget_graph = plotting.WidgetGraph(add_yruler=False)

        self.plot_item = {}
        self.plot_data = {}

        self.chunk_size = 1000

        self.vlayout_display = pg.GraphicsLayoutWidget()
        self.vlayout_display.addItem(self.widget_frame, row=0, col=0)
        self.vlayout_display.addItem(self.widget_face, row=0, col=1)
        self.vlayout_display.addItem(self.widget_graph, row=1, col=0, colspan=2)
        self.vlayout_display.ci.layout.setRowStretchFactor(0, 2)
        self.vlayout_display.ci.layout.setRowStretchFactor(1, 3)

        self.widget_r = QtWidgets.QWidget()

        self.vlayout_rs = QtWidgets.QVBoxLayout()
        self.flayout_se = QtWidgets.QFormLayout()

        self.widget_r.setLayout(self.vlayout_rs)

        self.addWidget(self.vlayout_display)
        self.addWidget(self.widget_r)

        self.cb_anal = QtWidgets.QCheckBox()
        self.button_video_open = QtWidgets.QPushButton(
            qta.icon("ph.video-camera-light"), "Open Video"
        )
        self.button_webcam_open = QtWidgets.QPushButton(
            qta.icon("mdi.webcam"), "Open Webcam"
        )

        self.button_start = QtWidgets.QPushButton(
            qta.icon("ph.chart-line-fill"), "Analyze"
        )
        self.button_stop = QtWidgets.QPushButton(qta.icon("ph.stop-light"), "Stop")

        self.bt_reset_graph = QtWidgets.QPushButton(
            qta.icon("msc.refresh"), "Reset Graph Y-Range"
        )
        self.bt_reset_graph.clicked.connect(lambda: self.widget_graph.setYRange(0, 1))

        self.bt_pause_resume = QtWidgets.QPushButton(
            qta.icon("ph.pause-light"), "Pause"
        )
        self.bt_pause_resume.setDisabled(True)

        self.combo_backend = QtWidgets.QComboBox()
        self.combo_backend.addItems(self.get("backend_options").keys())
        self.add_handler(
            "backend", self.combo_backend, mapper=self.get("backend_options")
        )
        self.combo_backend.currentTextChanged.connect(self.save_conf)

        self.pb_anal = QtWidgets.QProgressBar()
        self.skip_faces = QtWidgets.QSpinBox()
        self.skip_frame = QtWidgets.QSpinBox()

        self.feature_group = QtWidgets.QGroupBox("Features")
        self.feature_layout = QtWidgets.QVBoxLayout()
        self.feature_group.setLayout(self.feature_layout)
        self.feature_ear = FeatureCheckBox(features.EARFeature)
        self.add_handler("feature_ear", self.feature_ear)
        self.feature_ear.clicked.connect(self.save_conf)
        self.feature_ear.clicked.connect(self.set_features)

        self.feature_checkboxes = [self.feature_ear]

        self.feature_layout.addWidget(self.feature_ear)

        self.vlayout_rs.addLayout(self.flayout_se)

        open_l = QtWidgets.QHBoxLayout()
        open_l.addWidget(self.button_video_open)
        open_l.addWidget(self.button_webcam_open)
        self.flayout_se.addRow(open_l)
        self.flayout_se.addRow(self.button_start)
        self.flayout_se.addRow(self.bt_pause_resume)
        self.flayout_se.addRow(self.button_stop)
        self.flayout_se.addRow(self.feature_group)
        self.flayout_se.addRow("Graph Update Delay:", self.skip_frame)
        self.flayout_se.addRow("Backend", self.combo_backend)
        self.flayout_se.addRow("Face Detection Skip:", self.skip_faces)
        self.flayout_se.addRow(self.pb_anal)
        self.flayout_se.addRow(self.bt_reset_graph)

        self.features: List[Type[features.Feature]] = []

        self.ea = analyser.LandmarkAnalyser()
        self.ea.register_hooks(self)

        self.button_video_open.clicked.connect(self.load_video)
        self.button_webcam_open.clicked.connect(self.load_webcam)

        self.button_start.clicked.connect(self.start)
        self.button_stop.clicked.connect(self.stop)
        self.bt_pause_resume.clicked.connect(self.ea.toggle_pause)

        self.skip_faces.setRange(3, 20)
        self.skip_faces.setValue(5)
        self.skip_faces.valueChanged.connect(self.ea.set_skip_count)

        self.skip_frame.setRange(1, 20)
        self.skip_frame.setValue(5)

        # disable analyse button and check box
        self.button_start.setDisabled(True)
        self.cb_anal.setDisabled(True)
        self.button_stop.setDisabled(True)

        self.setStretchFactor(0, 6)
        self.setStretchFactor(1, 4)

        self.set_features()

    def setup_graph(self) -> None:
        logger.info("Setup graph for all features to plot", features=self.features)
        self.widget_graph.clear()
        self.update_count = 0
        for k in self.plot_item:
            self.widget_graph.removeItem(k)

        self.plot_item.clear()
        self.plot_data.clear()

        # compute the x ticks for the graph based on the fps and the chunk size
        fps = self.ea.get_fps()
        logger.info("Video fps:", fps=fps)
        x_ticks = np.arange(0, self.chunk_size, fps)
        x_ticks_lab = [str(-(x / fps)) for x in np.flip(x_ticks)]

        x_axis = self.widget_graph.getAxis("bottom")
        x_axis.setLabel("Internal Video Time [s]")
        x_axis.setTicks(
            [
                [(x, xl) for x, xl in zip(x_ticks[::10], x_ticks_lab[::10])],
                [(x, xl) for x, xl in zip(x_ticks, x_ticks_lab)],
            ]
        )

        for feature in self.features:
            for k, v in feature.plot_info.items():
                self.plot_item[k] = self.widget_graph.add_curve(**v)
                self.plot_data[k] = np.zeros(self.chunk_size)

    def set_features(self) -> None:
        self.features.clear()
        for c in self.feature_checkboxes:
            if c.isChecked():
                self.features.append(c.feature)
        logger.info("Set features", features=self.features)

    def start(self) -> None:
        self.ea.set_resource_path(self.video_file_path or 0)
        self.setup_graph()
        self.ea.set_settings(backend=self.get("backend"))
        self.ea.set_features(self.features)
        self.ea.start()

    def stop(self) -> None:
        self.ea.stop()

    @analyser.hookimpl
    def started(self):
        self.button_video_open.setDisabled(True)
        self.button_webcam_open.setDisabled(True)

        self.button_start.setDisabled(True)
        self.button_stop.setDisabled(False)
        self.cb_anal.setDisabled(True)
        self.feature_group.setDisabled(True)
        self.combo_backend.setDisabled(True)
        self.bt_pause_resume.setDisabled(False)

        if self.combo_backend.currentText() == "mediapipe":
            self.skip_faces.setDisabled(True)

    @analyser.hookimpl
    def paused(self):
        self.bt_pause_resume.setText("Resume")
        self.bt_pause_resume.setIcon(qta.icon("ph.play-light"))

    @analyser.hookimpl
    def resumed(self):
        self.bt_pause_resume.setText("Pause")
        self.bt_pause_resume.setIcon(qta.icon("ph.pause-light"))

    @analyser.hookimpl
    def finished(self):
        self.save_results()

        self.button_video_open.setDisabled(False)
        self.button_webcam_open.setDisabled(False)

        self.button_start.setText("Analyze Video")
        self.button_start.setDisabled(False)
        self.button_stop.setDisabled(True)
        self.feature_group.setDisabled(False)
        self.combo_backend.setDisabled(False)
        self.cb_anal.setDisabled(False)

        # reset the pause/resume button
        self.bt_pause_resume.setDisabled(True)
        self.bt_pause_resume.setText("Pause")

        if self.combo_backend.currentText() == "mediapipe":
            self.skip_faces.setDisabled(False)

    @analyser.hookimpl
    def updated_display(self, image: np.ndarray, face: np.ndarray):
        self.widget_frame.set_image(image)
        self.widget_face.set_image(face)

    @analyser.hookimpl
    def processed_percentage(self, percentage: int):
        self.pb_anal.setValue(percentage)

    @analyser.hookimpl
    def updated_feature(self, feature_data: OrderedDict[str, Any]) -> None:
        self.update_count += 1
        for feat in self.features:
            f_name = feat.__name__
            if f_name in feature_data:
                for k in feat.plot_info:
                    val = getattr(feature_data[f_name], k)
                    self.plot_data[k][:-1] = self.plot_data[k][1:]
                    self.plot_data[k][-1] = val

                    if self.update_count % self.skip_frame.value() == 0:
                        self.plot_item[k].setData(self.plot_data[k])

    def load_video(self):
        logger.info("Open File Dialog", widget=self)
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select video file",
            ".",
            "Video Files (*.mp4 *.flv *.ts *.mts *.avi *.mov)",
            options=QtWidgets.QFileDialog.DontUseNativeDialog,
        )

        if fileName != "":
            logger.info("Video file selected", file_name=fileName)
            self.video_file_path = pathlib.Path(fileName)

            self.button_start.setDisabled(False)

            self.load_cleanup()
        else:
            logger.info("Open File Dialog canceled")
            self.button_start.setDisabled(True)

    def load_webcam(self):
        logger.info("Open Webcam", widget=self)
        self.video_file_path = None
        self.button_start.setDisabled(False)
        self.skip_frame.setValue(1)
        self.load_cleanup()
        self.start()

    def load_cleanup(self) -> None:
        self.widget_frame.set_image(np.ones((100, 100, 3), dtype=np.uint8) * 255)
        self.widget_face.set_image(np.ones((100, 100, 3), dtype=np.uint8) * 255)
        self.widget_graph.clear()

    def save_results(self) -> None:
        logger.info("Save Results Dialog", widget=self)
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if self.video_file_path is not None:
            parent = self.video_file_path.parent
            file_name = self.video_file_path.stem
        else:
            parent = pathlib.Path.home()
            file_name = "jefapato_webcam"

        result_path = parent / (file_name + f"_{ts}.csv")

        with open(result_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            header = self.ea.get_header()
            writer.writerow(header)
            writer.writerows(self.ea)

        logger.info("Results saved", file_name=result_path)

    def shut_down(self) -> None:
        logger.info("Shutdown", state="starting", widget=self)
        self.stop()
        logger.info("Shutdown", state="finished", widget=self)