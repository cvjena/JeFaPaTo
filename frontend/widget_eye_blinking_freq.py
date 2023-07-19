from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
import qtawesome as qta
import structlog
from qtpy import QtCore, QtGui, QtWidgets
from tabulate import tabulate

from jefapato.methods import blinking
from frontend import plotting, config
logger = structlog.get_logger()

DOWNSAMPLE_FACTOR = 8


class QHLine(QtWidgets.QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)

class QVLine(QtWidgets.QFrame):
    def __init__(self):
        super(QVLine, self).__init__()
        self.setFrameShape(QtWidgets.QFrame.VLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)



class CollapsibleBox(QtWidgets.QWidget):
    # https://stackoverflow.com/questions/52615115/how-to-create-collapsible-box-in-pyqt/52617714#52617714
    def __init__(self, title="", parent=None):
        super(CollapsibleBox, self).__init__(parent)

        self.toggle_button = QtWidgets.QToolButton(text=title, checkable=True, checked=False)
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(QtCore.Qt.RightArrow)
        self.toggle_button.pressed.connect(self.on_pressed)

        self.toggle_animation = QtCore.QParallelAnimationGroup(self)

        self.content_area = QtWidgets.QScrollArea(maximumHeight=0, minimumHeight=0)
        self.content_area.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.content_area.setFrameShape(QtWidgets.QFrame.NoFrame)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toggle_button)
        lay.addWidget(self.content_area)

        self.toggle_animation.addAnimation(QtCore.QPropertyAnimation(self, b"minimumHeight"))
        self.toggle_animation.addAnimation(QtCore.QPropertyAnimation(self, b"maximumHeight"))
        self.toggle_animation.addAnimation(QtCore.QPropertyAnimation(self.content_area, b"maximumHeight"))

    @QtCore.Slot()
    def on_pressed(self):
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(QtCore.Qt.DownArrow if not checked else QtCore.Qt.RightArrow)
        self.toggle_animation.setDirection(QtCore.QAbstractAnimation.Forward if not checked else QtCore.QAbstractAnimation.Backward)
        self.toggle_animation.start()

    def setContentLayout(self, layout):
        lay = self.content_area.layout()
        del lay
        self.content_area.setLayout(layout)
        collapsed_height = self.sizeHint().height() - self.content_area.maximumHeight()
        content_height = layout.sizeHint().height()
        for i in range(self.toggle_animation.animationCount()):
            animation = self.toggle_animation.animationAt(i)
            animation.setDuration(500)
            animation.setStartValue(collapsed_height)
            animation.setEndValue(collapsed_height + content_height)

        content_animation = self.toggle_animation.animationAt(self.toggle_animation.animationCount() - 1)
        content_animation.setDuration(500)
        content_animation.setStartValue(0)
        content_animation.setEndValue(content_height)


def _get_QGroupBox(self):
    return self.isChecked()


def _set_QGroupBox(self, val):
    self.setChecked(val)


def _event_QGroupBox(self):
    return self.clicked


def to_MM_SS(value):
    return f"{int(value / 60):02d}:{int(value % 60):02d}"

def sec_to_min(seconds: float) -> str:
    minutes = int(seconds / 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def to_qt_row(row: pd.Series) -> list:
    return [QtGui.QStandardItem(str(row[c])) for c in row.index]

class WidgetEyeBlinkingFreq(QtWidgets.QSplitter, config.Config):
    updated = QtCore.Signal(int)

    def __init__(self, parent=None):
        config.Config.__init__(self, prefix="ear")
        QtWidgets.QSplitter.__init__(self, parent=parent)

        logger.info("Initializing EyeBlinkingFreq widget")
        self.result_text: str = ""

        self.x_lim_max = 1000
        self.raw_ear_l: np.ndarray | None = None
        self.raw_ear_r: np.ndarray | None = None

        self.blinking_l: pd.DataFrame | None = None
        self.blinking_r: pd.DataFrame | None = None

        self.lines: list = []
        self.file: Path | None = None
        self.data_frame: pd.DataFrame | None = None
        self.data_frame_columns: list[str] = []

        self.graph = plotting.WidgetGraph(x_lim_max=self.x_lim_max)
        self.plot_curve_ear_l = self.graph.add_curve({"color": "#00F", "width": 2}) # TODO add correct colors...
        self.plot_curve_ear_r = self.graph.add_curve({"color": "#F00", "width": 2}) # TODO add correct colors...
        self.plot_scatter_blinks_l = self.graph.add_scatter()
        self.plot_scatter_blinks_r = self.graph.add_scatter()

        # UI elements 
        self.add_hooks(QtWidgets.QGroupBox, (_get_QGroupBox, _set_QGroupBox, _event_QGroupBox))
        self.setOrientation(QtCore.Qt.Horizontal)

        self.setAcceptDrops(True)
        # Create the main layouts of the interface
        widget_content = QtWidgets.QWidget()
        self.layout_content = QtWidgets.QVBoxLayout()
        widget_content.setLayout(self.layout_content)

        widget_settings = QtWidgets.QWidget()
        self.layout_settings = QtWidgets.QVBoxLayout()
        widget_settings.setLayout(self.layout_settings)

        self.addWidget(widget_content)
        self.addWidget(widget_settings)

        self.setStretchFactor(0, 6)
        self.setStretchFactor(1, 4)

        # Create the specific widgets for the content layout
        self.tab_widget_results = QtWidgets.QTabWidget()

        # upper main content is a tab widget with the tables and text information
        # first tabe is the tables with the results
        self.splitter_table = QtWidgets.QSplitter(QtCore.Qt.Horizontal, parent=self)
        self.model_l = QtGui.QStandardItemModel(self)
        self.model_r = QtGui.QStandardItemModel(self)

        self.table_l = QtWidgets.QTableView()
        self.table_l.setModel(self.model_l)
        self.table_r = QtWidgets.QTableView()
        self.table_r.setModel(self.model_r)

        self.splitter_table.addWidget(self.table_l)
        self.splitter_table.addWidget(self.table_r)

        # second tab is the text information
        self.te_results_g = QtWidgets.QTextEdit()
        self.te_results_g.setFontFamily("mono")
        self.te_results_g.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.NoWrap)

        self.tab_widget_results.addTab(self.splitter_table, "Table Results")
        self.tab_widget_results.addTab(self.te_results_g, "Analysis Results")

        # lower main content is a graph
        self.graph_layout = pg.GraphicsLayoutWidget()
        self.graph.getViewBox().enableAutoRange(enable=False)
        self.graph.setYRange(0, 1)
        self.graph_layout.addItem(self.graph)

        # Create the specific widgets for the settings layout
        self.layout_content.addWidget(self.tab_widget_results)
        self.layout_content.addWidget(self.graph_layout)

        # Create the specific widgets for the settings layout
        # algorithm specific settings
        self.btn_load = QtWidgets.QPushButton(qta.icon("ph.folder-open-light"), "Open CSV File")
        self.btn_anal = QtWidgets.QPushButton(qta.icon("ph.chart-line-fill"), "Extract Blinks")
        self.btn_summ = QtWidgets.QPushButton(qta.icon("ph.info-light"), "Compute Summary")
        self.btn_eprt = QtWidgets.QPushButton(qta.icon("ph.export-light"), "Save")

        self.la_current_file = QtWidgets.QLabel("File: No file loaded")
        self.la_current_file.setWordWrap(True)

        self.comb_ear_l = QtWidgets.QComboBox()
        self.comb_ear_r = QtWidgets.QComboBox()
        self.comb_ear_l.currentIndexChanged.connect(self.select_column_left)
        self.comb_ear_r.currentIndexChanged.connect(self.select_column_right)

        self.progress = self.parent().progress_bar
        self.btn_load.clicked.connect(self.load_dialog)
        self.btn_anal.clicked.connect(self.extract_blinks)
        self.btn_summ.clicked.connect(self.compute_summary)
        self.btn_eprt.clicked.connect(self.save_results)

        # algorithm settings box
        self.box_settings = CollapsibleBox("Algorithm Settings")
        self.set_algo = QtWidgets.QFormLayout()

        le_th_l = QtWidgets.QLineEdit()
        le_th_r = QtWidgets.QLineEdit()
        le_fps = QtWidgets.QLineEdit()
        le_distance = QtWidgets.QLineEdit()
        le_prominence = QtWidgets.QLineEdit()
        le_width_min = QtWidgets.QLineEdit()
        le_width_max = QtWidgets.QLineEdit()

        MAPPER_FLOAT_STR = (lambda x: float(x), lambda x: str(x))
        MAPPTER_INT_STR = (lambda x: int(x), lambda x: str(x))

        self.add_handler("threshold_l", le_th_l, mapper=MAPPER_FLOAT_STR, default=0.16)
        self.add_handler("threshold_r", le_th_r, mapper=MAPPER_FLOAT_STR, default=0.16)
        self.add_handler("fps", le_fps, mapper=MAPPTER_INT_STR, default=240)
        self.add_handler("min_dist", le_distance, mapper=MAPPTER_INT_STR, default=50)
        self.add_handler("min_prominence", le_prominence, mapper=MAPPER_FLOAT_STR, default=0.1)
        self.add_handler("min_width", le_width_min, mapper=MAPPTER_INT_STR, default=80)
        self.add_handler("max_width", le_width_max, mapper=MAPPTER_INT_STR, default=500)

        le_fps.setValidator(QtGui.QIntValidator())
        le_width_min.setValidator(QtGui.QIntValidator())
        le_width_max.setValidator(QtGui.QIntValidator())

        self.set_algo.addRow("Threshold Left", le_th_l)
        self.set_algo.addRow("Threshold Right", le_th_r)
        self.set_algo.addRow("FPS", le_fps)
        self.set_algo.addRow("Min Distance", le_distance)
        self.set_algo.addRow("Min Prominence", le_prominence)
        self.set_algo.addRow("Min Width", le_width_min)
        self.set_algo.addRow("Max Width", le_width_max)

        box_smooth = QtWidgets.QGroupBox("Smoothing")
        box_smooth.setCheckable(True)
        self.add_handler("smooth", box_smooth)
        box_smooth_layout = QtWidgets.QFormLayout()
        box_smooth.setLayout(box_smooth_layout)

        le_smooth_size = QtWidgets.QLineEdit()
        le_smooth_poly = QtWidgets.QLineEdit()
        self.add_handler("smooth_size", le_smooth_size, mapper=MAPPTER_INT_STR, default=91)
        self.add_handler("smooth_poly", le_smooth_poly, mapper=MAPPTER_INT_STR, default=5)
        le_smooth_size.setValidator(QtGui.QIntValidator())
        le_smooth_poly.setValidator(QtGui.QIntValidator())

        box_smooth_layout.addRow("Polynomial Degree", le_smooth_poly)
        box_smooth_layout.addRow("Window Size", le_smooth_size)

        self.set_algo.addRow(box_smooth)

        # Visual Settings #
        self.box_visuals = CollapsibleBox("Visual Settings")
        self.set_visuals = QtWidgets.QFormLayout()

        cb_as_time = QtWidgets.QCheckBox()
        cb_width_height = QtWidgets.QCheckBox()
        cb_simple_draw = QtWidgets.QCheckBox()
        btn_reset_graph = QtWidgets.QPushButton(qta.icon("msc.refresh"), "Reset Graph Y Range")

        self.fps_box = QtWidgets.QGroupBox()
        self.radio_30 = QtWidgets.QRadioButton("30")
        self.radio_240 = QtWidgets.QRadioButton("240")
        self.radio_240.setChecked(True)
        
        self.radio_30.toggled.connect(self.compute_graph_axis)

        group_box_layout = QtWidgets.QHBoxLayout()
        self.fps_box.setLayout(group_box_layout)
        group_box_layout.addWidget(self.radio_30)
        group_box_layout.addWidget(self.radio_240)

        self.add_handler("as_time", cb_as_time)
        self.add_handler("draw_width_height", cb_width_height)
        self.add_handler("vis_downsample", cb_simple_draw)

        cb_as_time.clicked.connect(self.compute_graph_axis)
        cb_simple_draw.clicked.connect(lambda _: self.plot_data())
        cb_width_height.clicked.connect(lambda _: self.plot_data())
        btn_reset_graph.clicked.connect(lambda: self.graph.setYRange(0, 1))

        self.set_visuals.addRow("FPS", self.fps_box)
        self.set_visuals.addRow("X-Axis As Time", cb_as_time)
        self.set_visuals.addRow("Draw Width/Height", cb_width_height)
        self.set_visuals.addRow("Simple Draw", cb_simple_draw)
        self.set_visuals.addRow(btn_reset_graph)

        self.format_export = QtWidgets.QComboBox()
        self.format_export.addItems(["CSV", "Excel"])
        self.format_export.setCurrentIndex(0)
        self.add_handler("export_format", self.format_export, default="Excel")

        # Layouting #
        self.box_settings.setContentLayout(self.set_algo)
        self.box_visuals.setContentLayout(self.set_visuals)
        self.box_settings.toggle_button.click()
        self.box_visuals.toggle_button.click()

        # add all things to the settings layout
        self.layout_settings.addWidget(self.btn_load)
        self.layout_settings.addWidget(self.la_current_file)
        self.layout_settings.addWidget(QHLine())
        
        self.layout_settings.addWidget(QtWidgets.QLabel("Left Eye"))
        self.layout_settings.addWidget(self.comb_ear_l)
        self.layout_settings.addWidget(QtWidgets.QLabel("Right Eye"))
        self.layout_settings.addWidget(self.comb_ear_r)
        self.layout_settings.addWidget(QHLine())

        self.layout_settings.addWidget(self.box_settings)
        self.layout_settings.addWidget(self.btn_anal)
        self.layout_settings.addWidget(self.btn_summ)
        self.layout_settings.addWidget(QHLine())
        self.layout_settings.addWidget(QtWidgets.QLabel("Export Format"))
        self.layout_settings.addWidget(self.format_export)
        self.layout_settings.addWidget(self.btn_eprt)
        self.layout_settings.addWidget(QHLine())
        self.layout_settings.addWidget(self.box_visuals)

        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.layout_settings.addWidget(spacer)

        logger.info("Initialized EyeBlinkingFreq widget")
        self.compute_graph_axis()

        self.disable_column_selection()
        self.disable_algorithm()
        self.disable_export()

    # loading of the file
    def load_dialog(self) -> None:
        logger.info("Open file dialo for loading CSV file")
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select csv file",
            ".",
            "Video Files (*.csv)",
            options=QtWidgets.QFileDialog.Option.DontUseNativeDialog,
        )

        if file_path == "":
            logger.info("No file selected")
            return
        
        logger.info("File selected via dialog", file=file_path)
        self.load_file(Path(file_path))

    def load_file(self, file_path: Path) -> None:
        self.progress.setValue(0)
    
        self.clear_on_new_file()
        ############################################################

        # load the file and parse it as data frame
        file_path = Path(file_path).resolve().absolute()
        self.file = file_path
        self.la_current_file.setText(f"File: {file_path.as_posix()}")
        self.data_frame = pd.read_csv(file_path.as_posix(), sep=",")
        self.data_frame_columns = list(self.data_frame.columns)
        self.data_frame_columns = list(filter(lambda x: x not in ["frame", "valid"], self.data_frame_columns))

        self.comb_ear_l.addItems(self.data_frame_columns)
        self.comb_ear_r.addItems(self.data_frame_columns)

        self.enable_column_selection()
        self.enable_algorithm()
        self.disable_export()

    def select_column_left(self, index: int) -> None:
        if self.data_frame is None or self.data_frame_columns is None:
            return
        
        self.raw_ear_l = self.data_frame[self.data_frame_columns[index]].to_numpy()
        self.update_plot_raw()
        self.disable_export()

    def select_column_right(self, index: int) -> None:
        if self.data_frame is None or self.data_frame_columns is None:
            return
        self.raw_ear_r = self.data_frame[self.data_frame_columns[index]].to_numpy()

        self.update_plot_raw()
        self.disable_export()

    def update_plot_raw(self) -> None:
        self.plot_curve_ear_l.clear()
        self.plot_curve_ear_r.clear()

        if self.raw_ear_l is not None:
            self.plot_curve_ear_l.setData(self.raw_ear_l)
        if self.raw_ear_r is not None:
            self.plot_curve_ear_r.setData(self.raw_ear_r)

        self.compute_graph_axis()

    def compute_graph_axis(self) -> None:
        logger.info("Compute graph x-axis")
        ds_factor = 1 if not self.get("vis_downsample") else DOWNSAMPLE_FACTOR

        # compute the x_lim_max
        if self.raw_ear_l is not None and self.raw_ear_r is not None:
            self.x_lim_max = max(len(self.raw_ear_l), len(self.raw_ear_r))

        self.graph.setLimits(xMin=0, xMax=self.x_lim_max)
        self.graph.setLimits(yMin=0, yMax=1)
        self.graph.setXRange(0, self.x_lim_max)
        self.graph.setYRange(0, 1)

        self.graph.getAxis("left").setLabel("EAR Score")
        x_axis = self.graph.getAxis("bottom")

        if self.get("as_time"):
            x_axis.setLabel("Time (MM:SS)")
            fps = 30 if self.radio_30.isChecked() else 240 # TODO make more general in the future
            x_ticks = np.arange(0, self.x_lim_max, fps)
            x_ticks_lab = [str(to_MM_SS(x // fps)) for x in x_ticks]

            # TODO add some miliseconds to the ticks
            x_axis.setTicks(
                [
                    [(x, xl) for x, xl in zip(x_ticks[::60], x_ticks_lab[::60])],
                    [(x, xl) for x, xl in zip(x_ticks[::10], x_ticks_lab[::10])],
                    [(x, xl) for x, xl in zip(x_ticks, x_ticks_lab)],
                ]
            )

        else:
            x_axis.setLabel("Frames (#)")
            x_ticks = np.arange(0, self.x_lim_max, 1)
            x_ticks_lab = [str(x * ds_factor) for x in x_ticks]

            x_axis.setTicks(
                [
                    [(x, xl) for x, xl in zip(x_ticks[::1000], x_ticks_lab[::1000])],
                    [(x, xl) for x, xl in zip(x_ticks[::100], x_ticks_lab[::100])],
                ]
            )

    # extraction of the blinks
    def extract_blinks(self) -> None:
        self.progress.setRange(0, 100)

        self.progress.setValue(0)
        self.compute_intervals()
        self.progress.setValue(60)

        self.plot_intervals()
        self.progress.setValue(80)

        self.tabulate_intervals()
        self.progress.setValue(100)

        self.enable_export()        

    def compute_intervals(self) -> None:
        assert self.data_frame is not None, "Somehow the data frame is None"
        assert self.data_frame_columns is not None, "Somehow the data frame columns are None"

        assert self.raw_ear_r is not None, "Somehow the raw ear right is None"
        assert self.raw_ear_l is not None, "Somehow the raw ear left is None"

        kwargs = {}
        kwargs["fps"] = self.get("fps")
        kwargs["distance"] = self.get("min_dist")
        kwargs["prominence"] = self.get("min_prominence")
        kwargs["width_min"] = self.get("min_width")
        kwargs["width_max"] = self.get("max_width")

        do_smoothing: bool = self.get("smooth") or False
        smooth_size: int = self.get("smooth_size") or 91
        smooth_poly: int = self.get("smooth_poly") or 5
        
        ear_l = blinking.smooth(self.raw_ear_l, smooth_size, smooth_poly) if do_smoothing else self.raw_ear_l
        ear_r = blinking.smooth(self.raw_ear_r, smooth_size, smooth_poly) if do_smoothing else self.raw_ear_r

        self.progress.setValue(40)

        threshold_l = self.get("threshold_l") or 0.16
        threshold_r = self.get("threshold_r") or 0.16

        self.blinking_l = blinking.peaks(ear_l, threshold=threshold_l, **kwargs)
        self.blinking_r = blinking.peaks(ear_r, threshold=threshold_r, **kwargs)

    def plot_intervals(self) -> None:
        if self.blinking_l is None or self.blinking_r is None:
            return
        # TODO add some kind of settings for the colors
        self.plot_scatter_blinks_l.clear()
        self.plot_scatter_blinks_r.clear()

        self.plot_scatter_blinks_l.setData(x=self.blinking_l["frame"].to_numpy(), y=self.blinking_l["score"].to_numpy(), pen={"color": "#00F", "width": 2})
        self.plot_scatter_blinks_r.setData(x=self.blinking_r["frame"].to_numpy(), y=self.blinking_r["score"].to_numpy(), pen={"color": "#F00", "width": 2})

    def tabulate_intervals(self) -> None:
        if self.blinking_l is None or self.blinking_r is None:
            return

        self.model_l.clear()
        self.model_r.clear()
        for _, row in self.blinking_l.iterrows():
            self.model_l.appendRow(to_qt_row(row))

        for _, row in self.blinking_r.iterrows():
            self.model_r.appendRow(to_qt_row(row))

        self.table_l.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table_r.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)

        self.model_l.setHorizontalHeaderLabels(list(self.blinking_l.columns))
        self.model_r.setHorizontalHeaderLabels(list(self.blinking_r.columns))

    def clear_on_new_file(self) -> None:
        self.raw_ear_l = None
        self.raw_ear_r = None
        self.blinking_l = None
        self.blinking_r = None
        self.data_frame = None
        self.data_frame_columns = []
        self.x_lim_max = 1000

        self.comb_ear_l.clear()
        self.comb_ear_r.clear()

        self.model_l.clear()
        self.model_r.clear()

        self.plot_curve_ear_l.clear()
        self.plot_curve_ear_r.clear()
        self.plot_scatter_blinks_l.clear()
        self.plot_scatter_blinks_r.clear()

        self.te_results_g.setText("")

        self.disable_column_selection()
        self.disable_algorithm()
        self.disable_export()

    # summary of the results
    def compute_summary(self) -> None:
        pass

    def print_results(
        self,
        blinking_l: pd.DataFrame,
        blinking_r: pd.DataFrame,
        **kwargs,
    ) -> None:
        # compute the video time (depending on the fps) of the peaks in the data frames
        # for 30 fps and the given fps in kwargs and added to the data frames
        # the columns are called "time30" and timeFPS and the values are in the form
        # MM:SS the time is computed from the frame of the data frame
        blinking_l["time30"] = blinking_l["frame"] / 30
        blinking_r["time30"] = blinking_r["frame"] / 30
        blinking_l["timeFPS"] = blinking_l["frame"] / kwargs["fps"]
        blinking_r["timeFPS"] = blinking_r["frame"] / kwargs["fps"]
        blinking_l["time30"] = blinking_l["time30"].apply(sec_to_min)
        blinking_r["time30"] = blinking_r["time30"].apply(sec_to_min)
        blinking_l["timeFPS"] = blinking_l["timeFPS"].apply(sec_to_min)
        blinking_r["timeFPS"] = blinking_r["timeFPS"].apply(sec_to_min)

        self._reset_result_text()

        self._add("===Video Info===")
        self._add(f"File: {self.file.as_posix()}")
        self._add(f"Runtime: {sec_to_min(len(self.ear_l) / kwargs['fps'])}")

        for k, v in kwargs.items():
            self._add(f"{k}: {v}")

        bins = np.arange(
            start=0,
            stop=len(self.ear_l) + 2 * 60 * kwargs["fps"],
            step=60 * kwargs["fps"],
        )
        hist_l, _ = np.histogram(blinking_l["frame"], bins=bins)
        hist_r, _ = np.histogram(blinking_r["frame"], bins=bins)

        self._add("===Blinking Info===")
        self._add(f"Blinks Per Minute L: {hist_l.tolist()}")
        self._add(f"Blinks Per Minute R: {hist_r.tolist()}")

        self._add(f"Avg. Freq. L: {np.mean(hist_l): 6.3f}")
        self._add(f"Avg. Freq. R: {np.mean(hist_r): 6.3f}")

        self._add(f"Avg. Freq. [wo/ last minute] L: {np.mean(hist_l[:-1]): 6.3f}")
        self._add(f"Avg. Freq. [wo/ last minute] R: {np.mean(hist_r[:-1]): 6.3f}")

        _mean = np.mean(blinking_l["width"])
        _std = np.std(blinking_l["width"])
        self._add(f"Avg. Len. L: {_mean: 6.3f} +/- {_std: 6.3f} [frames]")
        _mean /= kwargs["fps"]
        _std /= kwargs["fps"]
        self._add(f"Avg. Len. L: {_mean: 6.3f} +/- {_std: 6.3f} [s]")

        _mean = np.mean(blinking_r["width"])
        _std = np.std(blinking_r["width"])
        self._add(f"Avg. Len. R: {_mean: 6.3f} +/- {_std: 6.3f} [frames]")
        _mean /= kwargs["fps"]
        _std /= kwargs["fps"]
        self._add(f"Avg. Len. R: {_mean: 6.3f} +/- {_std: 6.3f} [s]")

        self._add()

        for index, (start, stop) in enumerate(zip(bins[:-2], bins[1:])):
            df = blinking_l[(blinking_l["frame"] >= start) & (blinking_l["frame"] < stop)]
            _mean = np.mean(df["width"])
            _std = np.std(df["width"])
            self._add(f"Minute {index:02d} L: {_mean: 6.3f} +/- {_std: 6.3f}[frames]")

            _mean /= kwargs["fps"]
            _std /= kwargs["fps"]
            self._add(f"Minute {index:02d} R: {_mean: 6.3f} +/- {_std: 6.3f}[s]")

        self._add()

        for index, (start, stop) in enumerate(zip(bins[:-2], bins[1:])):
            df = blinking_r[(blinking_r["frame"] >= start) & (blinking_r["frame"] < stop)]
            _mean = np.mean(df["width"])
            _std = np.std(df["width"])
            self._add(f"Minute {index:02d} R: {_mean: 6.3f} +/- {_std: 6.3f}[frames]")

            _mean /= kwargs["fps"]
            _std /= kwargs["fps"]
            self._add(f"Minute {index:02d} R: {_mean: 6.3f} +/- {_std: 6.3f}[s]")

        self._add("")
        self.progress.setValue(95)
        self._add("===Detail Left Info===")
        self._add(tabulate(blinking_l, headers="keys", tablefmt="github"))
        self._add("")
        self._add("===Detail Right Info===")
        self._add(tabulate(blinking_r, headers="keys", tablefmt="github"))
        self._set_result_text()

        self.fill_tables(blinking_l, blinking_r)

        self.blinking_l = blinking_l
        self.blinking_r = blinking_r

    def _reset_result_text(self) -> None:
        self.result_text = ""
        self.te_results_g.setText("")

    def _add(self, text: str = "") -> None:
        self.result_text += text + "\n"

    def _set_result_text(self) -> None:
        self.te_results_g.setText(self.result_text)

    # saving of the results
    def save_results(self) -> None:
        if self.data_frame is None or self.file is None:
            return
        
        if self.blinking_l is None or self.blinking_r is None:
            logger.error("No blinking results to save", widget=self)
            return

        # TODO add summary export        zs
        # file_info = self.file.parent / (self.file.stem + "_blinking_info.txt")
        # logger.info("Saving blinking results", file=file_info)
        # file_info.write_text(self.result_text)
        
        if self.format_export.currentText() == "CSV":
            self.blinking_l.to_csv(self.file.parent / (self.file.stem + "_blinking_l.csv"), index=False)
            self.blinking_r.to_csv(self.file.parent / (self.file.stem + "_blinking_r.csv"), index=False)
        elif self.format_export.currentText() == "Excel":
            exel_file = self.file.parent / (self.file.stem + "_blinking.xlsx")
            logger.info("Saving blinking results", file=exel_file)
            with pd.ExcelWriter(exel_file) as writer:
                self.blinking_l.to_excel(writer, sheet_name="Left")
                self.blinking_r.to_excel(writer, sheet_name="Right")
        else:
            # raise NotImplementedError("Export format not implemented")
            logger.error("Export format not implemented", widget=self)
        logger.info("Saving blinking finished")
    
    ## general widget functions
    def shut_down(self) -> None:
        # this widget doesn't have any shut down requirements
        self.save()

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
        if file.suffix.lower() == ".csv":
            self.load_file(file)
            return
        
        logger.info("User dropped invalid file", widget=self)

    ## enabling for logic flow
    def enable_column_selection(self) -> None:
        self.comb_ear_l.setEnabled(True)
        self.comb_ear_r.setEnabled(True)

    def disable_column_selection(self) -> None:
        self.comb_ear_l.setEnabled(False)
        self.comb_ear_r.setEnabled(False)

    def enable_algorithm(self) -> None:
        self.box_settings.setEnabled(True)
        self.btn_anal.setEnabled(True)

    def disable_algorithm(self) -> None:
        self.box_settings.setEnabled(False)
        self.btn_anal.setEnabled(False)

    def enable_export(self) -> None:
        self.btn_eprt.setEnabled(True)
        self.btn_summ.setEnabled(True)
        self.format_export.setEnabled(True)

    def disable_export(self) -> None:
        self.btn_eprt.setEnabled(False)
        self.btn_summ.setEnabled(False)
        self.format_export.setEnabled(False)