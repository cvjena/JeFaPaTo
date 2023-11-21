__all__ = ["LandmarkExtraction"]

from collections import OrderedDict
import csv
import datetime
from pathlib import Path
import sys
from typing import Any, Callable, Type

import numpy as np
import qtawesome as qta
import structlog
from plyer import notification
from PyQt6.QtCore import pyqtSignal
from qtpy import QtCore, QtGui, QtWidgets

from frontend import config, jwidgets
from jefapato import facial_features
from jefapato.facial_features import features

logger = structlog.get_logger()


class FeatureCheckBox(QtWidgets.QCheckBox):
    def __init__(self, feature_class: Type[features.Feature], **kwargs):
        super().__init__(**kwargs)
        self.feature_class = feature_class

        name = feature_class.__name__
        name = name.replace("Feature", "")
        name = name.replace("BS_", "")
        name = name.replace("Left", "")
        name = name.replace("Right", "")
        self.setText(name)

class FeatureGroupBox(QtWidgets.QGroupBox):
    def __init__(self, title: str, callbacks: list[Callable] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.setTitle(title)
        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.setCheckable(True)
        self.toggled.connect(self.on_toggle)

        self.feature_checkboxes: list[FeatureCheckBox] = []
        self.callsbacks = callbacks or []

    def add_feature(self, feature_class: Type[features.Feature]):
        check_box = FeatureCheckBox(feature_class)
        self.layout().addWidget(check_box)
        self.feature_checkboxes.append(check_box)

        for callback in self.callsbacks:
            # this is a hacky workaround but currently the only way to do it
            if callback.__name__ == "add_handler":
                callback(check_box.feature_class.__name__, check_box, default=False)
            else:
                check_box.clicked.connect(callback)

    def on_toggle(self, on: bool):
        for box in self.sender().findChildren(QtWidgets.QCheckBox): # type: ignore
            box = box # type: FeatureCheckBox
            box.setChecked(on)
            box.setEnabled(True)

    def get_features(self) -> list[FeatureCheckBox]:
        return [box for box in self.feature_checkboxes if box.isChecked()]

class BlendShapeFeatureGroupBox(QtWidgets.QGroupBox):
    def __init__(self, callbacks: list[Callable] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.callsbacks = callbacks or []
        self.setTitle("Blend Shape Features")

        self.features_left  = FeatureGroupBox(title="Left",  callbacks=callbacks)
        self.features_right = FeatureGroupBox(title="Right", callbacks=callbacks)
        self.features_whole = FeatureGroupBox(title="Whole", callbacks=callbacks)

        self.setLayout(QtWidgets.QVBoxLayout())

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        # dont maximize the scroll area
        scroll.setSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding, QtWidgets.QSizePolicy.Policy.MinimumExpanding)
        self.layout().addWidget(scroll)

        temp_widget = QtWidgets.QWidget()
        temp_layout = QtWidgets.QHBoxLayout()
        temp_widget.setLayout(temp_layout)
        temp_layout.addWidget(self.features_whole, stretch=1)
        temp_layout.addWidget(self.features_right, stretch=1)
        temp_layout.addWidget(self.features_left, stretch=1)

        scroll.setWidget(temp_widget)

    def add_feature(self, feature_class: Type[features.Blendshape]):
        if "Left" in feature_class.__name__:
            self.features_left.add_feature(feature_class)
        elif "Right" in feature_class.__name__:
            self.features_right.add_feature(feature_class)
        else:
            self.features_whole.add_feature(feature_class)

    def get_features(self) -> list[FeatureCheckBox]:
        return self.features_left.get_features() + self.features_right.get_features() + self.features_whole.get_features()

class JeFaPaToSignalThread(QtCore.QThread):
    sig_updated_display = pyqtSignal(np.ndarray)
    sig_updated_feature = pyqtSignal(dict)
    sig_processed_percentage = pyqtSignal(float)
    sig_started = pyqtSignal()
    sig_paused = pyqtSignal()
    sig_resumed = pyqtSignal()
    sig_finished = pyqtSignal()
    
    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent=parent)
        self.parent = parent

    def run(self):
        while True:
            if self.stopped:
                return
    def stop(self):
        self.stopped = True

    @facial_features.FaceAnalyzer.hookimpl
    def updated_display(self, image: np.ndarray):
        self.sig_updated_display.emit(image)

    @facial_features.FaceAnalyzer.hookimpl
    def updated_feature(self, feature_data: OrderedDict[str, Any]) -> None:
        self.sig_updated_feature.emit(feature_data)
        
    @facial_features.FaceAnalyzer.hookimpl
    def processed_percentage(self, percentage: float) -> None:
        self.sig_processed_percentage.emit(percentage)
    
    @facial_features.FaceAnalyzer.hookimpl
    def started(self) -> None:
        self.sig_started.emit()
    
    @facial_features.FaceAnalyzer.hookimpl
    def paused(self) -> None:
        self.sig_paused.emit()
    
    @facial_features.FaceAnalyzer.hookimpl
    def resumed(self) -> None:
        self.sig_resumed.emit()

    @facial_features.FaceAnalyzer.hookimpl
    def finished(self) -> None:
        self.sig_finished.emit()

class LandmarkExtraction(QtWidgets.QSplitter, config.Config):
    updated = pyqtSignal(int) 

    def __init__(self, parent):
        config.Config.__init__(self, prefix="landmarks")
        QtWidgets.QSplitter.__init__(self, parent=parent)

        self.used_features_classes: list[Type[features.Feature]] = [features.BS_Valid]
        self.video_resource: Path | int | None = None
        self.plot_item = {}
        self.plot_data = {}
        self.chunk_size = 1000
        self.ea = facial_features.FaceAnalyzer()

        # UI elements
        self.setAcceptDrops(True)
        self.main_window = parent # type: QtWidgets.QMainWindow

        self.widget_frame = jwidgets.JVideoFaceSelection()
        self.widget_graph = jwidgets.JGraph(add_yruler=False)
        
        self.widget_graph.setLimits(xMin=0, xMax=self.chunk_size, yMin=0, yMax=1)

        layout_activity = QtWidgets.QVBoxLayout()
        layout_activity.addWidget(self.widget_frame, stretch=1)
        layout_activity.addWidget(self.widget_graph, stretch=1)
        
        widget_l = QtWidgets.QWidget()
        widget_r = QtWidgets.QWidget()
        widget_r.setFixedWidth(600)
        self.vlayout_rs = QtWidgets.QVBoxLayout()
        self.flayout_se = QtWidgets.QFormLayout()

        widget_l.setLayout(layout_activity)
        widget_r.setLayout(self.vlayout_rs)

        self.addWidget(widget_l)
        self.addWidget(widget_r)

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

        self.pb_anal = self.parent().progress_bar # type: ignore # TODO: fix this as JeFaPaTo cannot be imported from here...
        self.skip_frame = QtWidgets.QSpinBox()

        self.auto_save = QtWidgets.QCheckBox("Auto-Save")
        self.auto_save.setToolTip("Save the extracted data automatically after the analysis is finished.")

        self.use_bbox = QtWidgets.QCheckBox("Use Bounding Box")
        self.use_bbox.setToolTip("Use the bounding box to extract the landmarks.")

        self.add_handler("auto_save", self.auto_save, default=True)
        self.add_handler("use_bbox", self.use_bbox, default=True)
        self.add_handler("auto_find_face", self.widget_frame.cb_auto_find, default=True)

        self.feature_group = FeatureGroupBox("Facial Features", [self.set_features, self.add_handler])
        self.feature_group.add_feature(features.EAR2D6)
        self.feature_group.add_feature(features.EAR3D6)
        self.feature_group.add_feature(features.Landmarks478)
        self.feature_group.add_feature(features.Landmarks68)

        self.blends_shape_group = BlendShapeFeatureGroupBox(callbacks=[self.set_features, self.add_handler])
        for blendshape in features.BLENDSHAPES:
            self.blends_shape_group.add_feature(blendshape)

        self.vlayout_rs.addLayout(self.flayout_se)

        open_l = QtWidgets.QHBoxLayout()
        open_l.addWidget(self.button_video_open)
        # open_l.addWidget(self.button_webcam_open)
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
        self.flayout_se.addRow(self.use_bbox)
        self.flayout_se.addRow(self.widget_frame.cb_auto_find)

        # add two labels to the bottom row of the main layout
        self.la_input = QtWidgets.QLabel("Loading: ### frame/s")
        self.la_proce = QtWidgets.QLabel("Processing: ### frames/s")
        self.la_input.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.la_proce.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

        self.main_window.statusBar().addWidget(self.la_input)
        self.main_window.statusBar().addWidget(self.la_proce)

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
        
        self.jefapato_signal_thread = JeFaPaToSignalThread(self)
        self.ea.register_hooks(self.jefapato_signal_thread)
        
        self.jefapato_signal_thread.sig_updated_display.connect(self.sig_updated_display)
        self.jefapato_signal_thread.sig_updated_feature.connect(self.sig_updated_feature)
        self.jefapato_signal_thread.sig_processed_percentage.connect(self.sig_processed_percentage)
        self.jefapato_signal_thread.sig_started.connect(self.sig_started)
        self.jefapato_signal_thread.sig_paused.connect(self.sig_paused)
        self.jefapato_signal_thread.sig_resumed.connect(self.sig_resumed)
        self.jefapato_signal_thread.sig_finished.connect(self.sig_finished)
        self.jefapato_signal_thread.start()
        
    def setup_graph(self) -> None:
        logger.info("Setup graph for all features to plot", features=len(self.used_features_classes))
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
        self.used_features_classes.extend([f.feature_class for f in self.feature_group.get_features()])
        self.used_features_classes.extend([f.feature_class for f in self.blends_shape_group.get_features()])
        self.used_features_classes.append(features.BS_Valid)
        # logger.info("Set features", features=self.used_features_classes)

    def start(self) -> None:
        # if self.video_resource is not None:
        #     self.set_resource(self.video_resource)
        self.setup_graph()
        self.ea.set_features(self.used_features_classes)
        
        rect = self.widget_frame.get_roi_rect() if self.use_bbox.isChecked() else None        
        self.ea.clean_start(rect)

    def stop(self) -> None:
        self.ea.stop()
        self.jefapato_signal_thread.stop()
        self.jefapato_signal_thread.wait()

    def sig_started(self):
        self.button_video_open.setDisabled(True)
        self.button_webcam_open.setDisabled(True)

        self.button_start.setDisabled(True)
        self.button_stop.setDisabled(False)
        self.cb_anal.setDisabled(True)
        self.feature_group.setDisabled(True)
        self.blends_shape_group.setDisabled(True)
        self.bt_pause_resume.setDisabled(False)
        self.setAcceptDrops(False)
        self.widget_frame.set_interactive(False)

    def sig_paused(self):
        self.bt_pause_resume.setText("Resume")
        self.bt_pause_resume.setIcon(qta.icon("ph.play-light"))

    def sig_resumed(self):
        self.bt_pause_resume.setText("Pause")
        self.bt_pause_resume.setIcon(qta.icon("ph.pause-light"))

    def sig_finished(self):
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
        self.widget_frame.set_interactive(True)
        
        try:
            notification.notify(
                title="Analysis finished",
                message="The analysis has finished and the next video can be analyzed.",
                app_name="JeFaPaTo",
                timeout = 10,
            )
        except Exception as e:
            logger.error("Failed to send notification", error=e, os=sys.platform)

    def sig_updated_display(self, image: np.ndarray):
        self.widget_frame.set_image(image)

    def sig_processed_percentage(self, percentage: int):
        self.pb_anal.setValue(int(percentage))
        data_input, data_proce = self.ea.get_throughput()
        self.la_input.setText(f"Input: {data_input: 3d} frames/s")
        self.la_proce.setText(f"Processed: {data_proce: 3d} frames/s")

    def sig_updated_feature(self, feature_data: dict[str, Any]) -> None:
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
        self.widget_frame.cb_auto_find.setChecked(False)
        self.use_bbox.setChecked(False)

        old_resource = self.video_resource
        if self.set_resource(-1):
            self.start()
        elif old_resource is not None:
            self.set_resource(old_resource)

    def set_resource(self, resource: Path | int) -> bool:
        self.video_resource = resource
        self.button_start.setDisabled(False)

        if isinstance(self.video_resource, Path):
            self.la_current_file.setText(f"File: {str(self.video_resource.absolute())}")
        else:
            self.la_current_file.setText("File: Live Webcam Feed")

        success, frame = self.ea.prepare_video_resource(self.video_resource)
        if success:
            logger.info("Image was set", parent=self)
            self.widget_frame.set_selection_image(frame)
        return success


    def save_results(self) -> None:
        return
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
        self.save()
        logger.info("Shutdown", state="finished", widget=self)

    def dragEnterEvent(self, event: QtGui.QDropEvent):
        logger.info("User started dragging event", widget=self)
        if event.mimeData().hasUrls():
            event.accept()
            logger.info("User started dragging event with mime file", widget=self)
        else:
            event.ignore()
            logger.info("User started dragging event with invalid mime file", widget=self)

    def dropEvent(self, event: QtGui.QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls()]

        if len(files) > 1:
            logger.info("User dropped multiple files", widget=self)
        file = files[0]

        file = Path(file)
        if file.suffix.lower() not in [".mp4", ".flv", ".ts", ".mts", ".avi", ".mov"]:
            logger.info("User dropped invalid file", widget=self)
            return
        self.set_resource(Path(file)) 