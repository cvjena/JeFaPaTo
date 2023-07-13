__all__ = ["LandmarkExtraction"]

import csv
import datetime
from pathlib import Path
from typing import Any, Type, Callable

import numpy as np
import pyqtgraph as pg
import qtawesome as qta
import structlog
from qtpy import QtCore, QtGui, QtWidgets

from jefapato import config, facial_features, plotting
from jefapato.facial_features import features


logger = structlog.get_logger()


class FeatureCheckBox(QtWidgets.QCheckBox):
    def __init__(self, feature_class: Type[features.Feature], **kwargs):
        super().__init__(**kwargs)
        self.feature_class = feature_class

        name = feature_class.__name__
        name = name.replace("Feature", "")
        name = name.replace("BS_", "")
        self.setText(name)

class FeatureGroupBox(QtWidgets.QGroupBox):
    def __init__(self, callbacks: list[Callable] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.setTitle("Facial Features")
        self.setLayout(QtWidgets.QVBoxLayout())
        self.feature_checkboxes: list[FeatureCheckBox] = []
        self.callsbacks = callbacks or []

    def add_feature(self, feature_class: Type[features.Feature]):
        # TODO rename cb to something more descriptive and not be confused with callback
        cb = FeatureCheckBox(feature_class)
        self.layout().addWidget(cb)
        self.feature_checkboxes.append(cb)

        for callback in self.callsbacks:
            # this is a hacky workaround but currently the only way to do it
            if callback.__name__ == "add_handler":
                callback(cb.feature_class.__name__, cb)
            else:
                cb.clicked.connect(callback)

class BlendShapeFeatureGroupBox(FeatureGroupBox):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setTitle("Blend Shape Features")

        # have two vertical layouts, one for the left and one for the right
        self.layout_left = QtWidgets.QVBoxLayout()
        self.layout_right = QtWidgets.QVBoxLayout()

        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().addLayout(self.layout_left)
        self.layout().addLayout(self.layout_left)
        

    def add_feature(self, feature_class: Type[features.Blendshape]):
        cb = FeatureCheckBox(feature_class)
        self.feature_checkboxes.append(cb)

        if feature_class.side == "left":
            self.layout_left.addWidget(cb)
        elif feature_class.side == "right":
            self.layout_right.addWidget(cb)
        else:
            self.layout().addWidget(cb)

        for callback in self.callsbacks:
            # this is a hacky workaround but currently the only way to do it
            if callback.__name__ == "add_handler":
                callback(cb.feature_class.__name__, cb)
            else:
                cb.clicked.connect(callback)

class LandmarkExtraction(QtWidgets.QSplitter, config.Config):
    updated = QtCore.Signal(int)

    def __init__(self, parent=None):
        config.Config.__init__(self, prefix="landmarks")
        QtWidgets.QSplitter.__init__(self, parent=parent)

        self.setAcceptDrops(True)

        self.video_resource: Path | int | None = None

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
        self.button_video_open = QtWidgets.QPushButton(qta.icon("ph.video-camera-light"), "Open Video")
        self.button_webcam_open = QtWidgets.QPushButton(qta.icon("mdi.webcam"), "Open Webcam")

        self.button_start = QtWidgets.QPushButton(qta.icon("ph.chart-line-fill"), "Analyze")
        self.button_stop = QtWidgets.QPushButton(qta.icon("ph.stop-light"), "Stop")

        self.bt_reset_graph = QtWidgets.QPushButton(qta.icon("msc.refresh"), "Reset Graph Y-Range")
        self.bt_reset_graph.clicked.connect(lambda: self.widget_graph.setYRange(0, 1))

        self.bt_pause_resume = QtWidgets.QPushButton(qta.icon("ph.pause-light"), "Pause")
        self.bt_pause_resume.setDisabled(True)

        self.la_current_file = QtWidgets.QLabel("File: No file loaded")
        self.la_current_file.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.la_current_file.setWordWrap(True)

        self.pb_anal = self.parent().progress_bar
        self.skip_frame = QtWidgets.QSpinBox()
        self.auto_save = QtWidgets.QCheckBox("Auto-Save")
        self.auto_save.setChecked(True)
        self.auto_save.setToolTip("Save the extracted data automatically after the analysis is finished.")

        self.feature_group = FeatureGroupBox([self.save_conf, self.set_features, self.add_handler])
        self.feature_group.add_feature(features.EAR2D6)
        self.feature_group.add_feature(features.EAR3D6)

        self.blends_shape_group = BlendShapeFeatureGroupBox(callbacks=[self.save_conf, self.set_features, self.add_handler])
        for blendshape in features.BLENDSHAPES:
            self.blends_shape_group.add_feature(blendshape)

        self.vlayout_rs.addLayout(self.flayout_se)

        open_l = QtWidgets.QHBoxLayout()
        open_l.addWidget(self.button_video_open)
        open_l.addWidget(self.button_webcam_open)
        self.flayout_se.addRow(open_l)
        self.flayout_se.addRow(self.la_current_file)
        self.flayout_se.addRow(self.button_start)
        self.flayout_se.addRow(self.bt_pause_resume)
        self.flayout_se.addRow(self.button_stop)
        self.flayout_se.addRow(self.feature_group)
        self.flayout_se.addRow(self.blends_shape_group)
        self.flayout_se.addRow("Graph Update Delay:", self.skip_frame)
        self.flayout_se.addRow(self.bt_reset_graph)
        self.flayout_se.addRow(self.auto_save)

        # add two labels to the bottom row of the main layout
        self.la_input = QtWidgets.QLabel("Loading: ### frame/s")
        self.la_proce = QtWidgets.QLabel("Processing: ### frames/s")
        self.la_input.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.la_proce.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

        self.parent().statusBar().addWidget(self.la_input)
        self.parent().statusBar().addWidget(self.la_proce)

        self.used_features_classes: list[Type[features.Feature]] = []

        self.ea = facial_features.FaceAnalyzer()
        self.ea.register_hooks(self)

        self.button_video_open.clicked.connect(self.load_video)
        self.button_webcam_open.clicked.connect(self.load_webcam)

        self.button_start.clicked.connect(self.start)
        self.button_stop.clicked.connect(self.stop)
        self.bt_pause_resume.clicked.connect(self.ea.toggle_pause)

        self.skip_frame.setRange(1, 20)
        self.skip_frame.setValue(1)

        # disable analyse button and check box
        self.button_start.setDisabled(True)
        self.cb_anal.setDisabled(True)
        self.button_stop.setDisabled(True)

        self.setStretchFactor(0, 6)
        self.setStretchFactor(1, 4)

        self.set_features()

    def setup_graph(self) -> None:
        logger.info("Setup graph for all features to plot", features=self.used_features_classes)
        self.widget_graph.clear()
        self.update_count = 0
        for feature_name in self.plot_item:
            self.widget_graph.removeItem(feature_name)

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

        for feature_class in self.used_features_classes:
            for feature_name, feature_plot_settings in feature_class.plot_info.items():
                feature_name = f"{feature_class.__name__}_{feature_name}"
                self.plot_item[feature_name] = self.widget_graph.add_curve(**feature_plot_settings)
                self.plot_data[feature_name] = np.zeros(self.chunk_size)

    def set_features(self) -> None:
        self.used_features_classes.clear()
        for c in self.feature_group.feature_checkboxes:
            if c.isChecked():
                self.used_features_classes.append(c.feature_class)
        for c in self.blends_shape_group.feature_checkboxes:
            if c.isChecked():
                self.used_features_classes.append(c.feature_class) 

        logger.info("Set features", features=self.used_features_classes)

    def start(self) -> None:
        if self.video_resource is not None:
            self.set_resource(self.video_resource)
        self.setup_graph()
        self.ea.set_features(self.used_features_classes)
        self.ea.start()

    def stop(self) -> None:
        self.ea.stop()

    @facial_features.FaceAnalyzer.hookimpl
    def started(self):
        self.button_video_open.setDisabled(True)
        self.button_webcam_open.setDisabled(True)

        self.button_start.setDisabled(True)
        self.button_stop.setDisabled(False)
        self.cb_anal.setDisabled(True)
        self.feature_group.setDisabled(True)
        self.blends_shape_group.setDisabled(True)
        self.bt_pause_resume.setDisabled(False)
        self.setAcceptDrops(False)

    @facial_features.FaceAnalyzer.hookimpl
    def paused(self):
        self.bt_pause_resume.setText("Resume")
        self.bt_pause_resume.setIcon(qta.icon("ph.play-light"))

    @facial_features.FaceAnalyzer.hookimpl
    def resumed(self):
        self.bt_pause_resume.setText("Pause")
        self.bt_pause_resume.setIcon(qta.icon("ph.pause-light"))

    @facial_features.FaceAnalyzer.hookimpl
    def finished(self):
        self.save_results()

        self.button_video_open.setDisabled(False)
        self.button_webcam_open.setDisabled(False)

        self.button_start.setText("Analyze Video")
        self.button_start.setDisabled(False)
        self.button_stop.setDisabled(True)
        self.feature_group.setDisabled(False)
        self.blends_shape_group.setDisabled(False)
        self.cb_anal.setDisabled(False)

        # reset the pause/resume button
        self.bt_pause_resume.setDisabled(True)
        self.bt_pause_resume.setText("Pause")
        self.setAcceptDrops(True)

    @facial_features.FaceAnalyzer.hookimpl
    def updated_display(self, image: np.ndarray, face: np.ndarray):
        self.widget_frame.set_image(image)
        self.widget_face.set_image(face)

    @facial_features.FaceAnalyzer.hookimpl
    def processed_percentage(self, percentage: int):
        self.pb_anal.setValue(percentage)

        data_input, data_proce = self.ea.get_throughput()
        self.la_input.setText(f"Input: {data_input: 3d} frames/s")
        self.la_proce.setText(f"Processed: {data_proce: 3d} frames/s")

    @facial_features.FaceAnalyzer.hookimpl
    def updated_feature(self, feature_data: dict[str, Any]) -> None:
        self.update_count += 1
        for feature_class in self.used_features_classes:
            if feature_class.__name__ not in feature_data:
                continue
            for feature_name in feature_class.plot_info.keys():
                feature_value = getattr(feature_data[feature_class.__name__], feature_name)
                feature_name = f"{feature_class.__name__}_{feature_name}"
                self.plot_data[feature_name][:-1] = self.plot_data[feature_name][1:]
                self.plot_data[feature_name][-1] = feature_value

                if self.update_count % self.skip_frame.value() == 0:
                    self.plot_item[feature_name].setData(self.plot_data[feature_name])

    def load_video(self):
        logger.info("Open File Dialog", widget=self)
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent=self,
            caption="Select video file",
            directory=".",
            filter="Video Files (*.mp4 *.flv *.ts *.mts *.avi *.mov)",
            options=QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )

        if fileName == "":
            logger.info("Open File Dialog canceled")
            self.button_start.setDisabled(True)
            self.la_current_file.setText("File: None selected")
            self.video_resource = None
            return
        
        logger.info("Open File Dialog selected", file=fileName)
        self.set_resource(Path(fileName))

    def load_webcam(self):
        logger.info("Open Webcam", widget=self)
        self.set_resource(-1)

    def set_resource(self, resource: Path | int) -> None:
        self.video_resource = resource
        self.button_start.setDisabled(False)

        if isinstance(self.video_resource, Path):
            self.la_current_file.setText(f"File: {str(self.video_resource.absolute())}")
        else:
            self.la_current_file.setText("File: Live Webcam Feed")

        success, frame = self.ea.prepare_video_resource(self.video_resource)
        if success:
            self.widget_frame.set_image(frame)


    def save_results(self) -> None:
        logger.info("Save Results Dialog", widget=self)
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if isinstance(self.video_resource, Path) and self.auto_save.isChecked():
            parent = self.video_resource.parent
            file_name = self.video_resource.stem
        else:
            # open save dialog for folder
            parent = QtWidgets.QFileDialog.getExistingDirectory(parent=self, caption="Select Directory",directory=str(Path.home()))
            if parent == "":
                logger.info("Save Results Dialog canceled")
                return

            parent = Path(parent)
            if isinstance(self.video_resource, int):
                file_name = "jefapato_webcam"
            elif isinstance(self.video_resource, Path):
                file_name = self.video_resource.stem
            else:
                raise ValueError("Invalid video resource")

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

    def dragEnterEvent(self, event: QtGui.QDropEvent):
        logger.info("User started dragging event", widget=self)
        if event.mimeData().hasUrls():
            event.accept()
            logger.info("User started dragging event with mime file", widget=self)
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls()]

        if len(files) > 1:
            logger.info("User dropped multiple files", widget=self)
        file = files[0]

        file = Path(file)
        if file.suffix not in [".mp4", ".flv", ".ts", ".mts", ".avi", ".mov"]:
            logger.info("User dropped invalid file", widget=self)
            return
        self.set_resource(Path(file)) 