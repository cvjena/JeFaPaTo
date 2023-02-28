import pathlib

import numpy as np
import pandas as pd
import pyqtgraph as pg
import qtawesome as qta
import structlog
from qtpy import QtCore, QtGui, QtWidgets
from scipy import signal
from tabulate import tabulate

from jefapato import config, plotting
from jefapato.methods import blinking

logger = structlog.get_logger()

DOWNSAMPLE_FACTOR = 8


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


class WidgetEyeBlinkingFreq(QtWidgets.QSplitter, config.Config):
    updated = QtCore.Signal(int)

    def __init__(self, parent=None):
        config.Config.__init__(self, prefix="ear")
        QtWidgets.QSplitter.__init__(self, parent=parent)

        self.add_hooks(QtWidgets.QGroupBox, (_get_QGroupBox, _set_QGroupBox, _event_QGroupBox))

        self.setOrientation(QtCore.Qt.Horizontal)

        logger.info("Initializing EyeBlinkingFreq widget")

        self.result_text: str = ""

        self.x_lim_max = 1000
        self.x_lim_max_old = 1000

        # Create the main layouts of the interface
        widget_content = QtWidgets.QWidget()
        self.layout_content = QtWidgets.QVBoxLayout()
        widget_content.setLayout(self.layout_content)

        widget_settings = QtWidgets.QWidget()
        self.layout_settings = QtWidgets.QGridLayout()
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
        self.graph = plotting.WidgetGraph(x_lim_max=self.x_lim_max)

        self.graph.getViewBox().enableAutoRange(enable=False)
        self.graph.setYRange(0, 1)
        self.graph_layout.addItem(self.graph)

        # Create the specific widgets for the settings layout
        self.layout_content.addWidget(self.tab_widget_results)
        self.layout_content.addWidget(self.graph_layout)

        # Create the specific widgets for the settings layout
        # algorithm specific settings
        self.btn_load = QtWidgets.QPushButton(qta.icon("ph.folder-open-light"), "Open CSV File")
        self.btn_anal = QtWidgets.QPushButton(qta.icon("ph.chart-line-fill"), "Analyse")
        self.btn_eprt = QtWidgets.QPushButton(qta.icon("ph.export-light"), "Export")

        self.progress = self.parent().progress_bar

        self.btn_load.clicked.connect(self._load_csv)
        self.btn_anal.clicked.connect(self._analyse)
        self.btn_eprt.clicked.connect(self.save_results)

        # algorithm settings box
        self.box_settings = CollapsibleBox("Algorithm Settings")
        set_algo = QtWidgets.QFormLayout()

        le_th_l = QtWidgets.QLineEdit()
        le_th_r = QtWidgets.QLineEdit()
        le_fps = QtWidgets.QLineEdit()
        le_distance = QtWidgets.QLineEdit()
        le_prominence = QtWidgets.QLineEdit()
        le_width_min = QtWidgets.QLineEdit()
        le_width_max = QtWidgets.QLineEdit()

        self.add_handler("threshold_l", le_th_l)
        self.add_handler("threshold_r", le_th_r)
        self.add_handler("fps", le_fps)
        self.add_handler("min_dist", le_distance)
        self.add_handler("min_prominence", le_prominence)
        self.add_handler("min_width", le_width_min)
        self.add_handler("max_width", le_width_max)

        le_th_l.textChanged.connect(self.save_conf)
        le_th_r.textChanged.connect(self.save_conf)
        le_fps.textChanged.connect(self.save_conf)
        le_fps.textChanged.connect(self.compute_graph_axis)
        le_distance.textChanged.connect(self.save_conf)
        le_prominence.textChanged.connect(self.save_conf)
        le_width_min.textChanged.connect(self.save_conf)
        le_width_max.textChanged.connect(self.save_conf)

        le_fps.setValidator(QtGui.QIntValidator())
        le_width_min.setValidator(QtGui.QIntValidator())
        le_width_max.setValidator(QtGui.QIntValidator())

        set_algo.addRow("Threshold Left", le_th_l)
        set_algo.addRow("Threshold Right", le_th_r)
        set_algo.addRow("FPS", le_fps)
        set_algo.addRow("Min Distance", le_distance)
        set_algo.addRow("Min Prominence", le_prominence)
        set_algo.addRow("Min Width", le_width_min)
        set_algo.addRow("Max Width", le_width_max)

        box_smooth = QtWidgets.QGroupBox("Smoothing")
        box_smooth.setCheckable(True)
        self.add_handler("smooth", box_smooth)
        box_smooth_layout = QtWidgets.QFormLayout()
        box_smooth.setLayout(box_smooth_layout)

        le_smooth_size = QtWidgets.QLineEdit()
        le_smooth_poly = QtWidgets.QLineEdit()
        self.add_handler("smooth_size", le_smooth_size)
        self.add_handler("smooth_poly", le_smooth_poly)
        le_smooth_size.setValidator(QtGui.QIntValidator())
        le_smooth_poly.setValidator(QtGui.QIntValidator())
        le_smooth_poly.textChanged.connect(self.save_conf)
        le_smooth_size.textChanged.connect(self.save_conf)

        box_smooth_layout.addRow("Polynomial Degree", le_smooth_poly)
        box_smooth_layout.addRow("Window Size", le_smooth_size)

        set_algo.addRow(box_smooth)

        # Visual Settings #

        self.box_visuals = CollapsibleBox("Visual Settings")
        set_visuals = QtWidgets.QFormLayout()

        cb_as_time = QtWidgets.QCheckBox()
        cb_width_height = QtWidgets.QCheckBox()
        cb_simple_draw = QtWidgets.QCheckBox()
        btn_reset_graph = QtWidgets.QPushButton(qta.icon("msc.refresh"), "Reset Graph Y Range")

        self.add_handler("as_time", cb_as_time)
        self.add_handler("draw_width_height", cb_width_height)
        self.add_handler("vis_downsample", cb_simple_draw)

        cb_width_height.clicked.connect(self.save_conf)
        cb_as_time.clicked.connect(self.save_conf)
        cb_simple_draw.clicked.connect(self.save_conf)

        cb_as_time.clicked.connect(self.compute_graph_axis)
        cb_simple_draw.clicked.connect(lambda _: self.plot_data())
        cb_width_height.clicked.connect(lambda _: self.plot_data())
        btn_reset_graph.clicked.connect(lambda: self.graph.setYRange(0, 1))

        set_visuals.addRow("X-Axis As Time", cb_as_time)
        set_visuals.addRow("Draw Width/Height", cb_width_height)
        set_visuals.addRow("Simple Draw", cb_simple_draw)
        set_visuals.addRow(btn_reset_graph)

        # Layouting #
        self.box_settings.setContentLayout(set_algo)
        self.box_visuals.setContentLayout(set_visuals)
        self.box_settings.toggle_button.click()
        self.box_visuals.toggle_button.click()

        # add all things to the settings layout
        self.layout_settings.addWidget(self.btn_load, 0, 0, 1, 2)
        self.layout_settings.addWidget(self.btn_anal, 1, 0, 1, 1)
        self.layout_settings.addWidget(self.btn_eprt, 1, 1, 1, 1)
        self.layout_settings.addWidget(self.box_settings, 2, 0, 1, 2)
        self.layout_settings.addWidget(self.box_visuals, 3, 0, 1, 2)
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self.layout_settings.addWidget(spacer, self.layout_settings.rowCount(), 0)

        self.ear_l = None
        self.ear_r = None

        self.raw_ear_l = None
        self.raw_ear_r = None

        self.plot_ear_l = self.graph.add_curve({"color": "#00F", "width": 2})
        self.plot_ear_r = self.graph.add_curve({"color": "#F00", "width": 2})

        self.plot_peaks_l = self.graph.add_scatter()
        self.plot_peaks_r = self.graph.add_scatter()

        self.blinking_l = pd.DataFrame()
        self.blinking_r = pd.DataFrame()

        self.lines = list()
        self.file = None

        logger.info("Initialized EyeBlinkingFreq widget")

        self.compute_graph_axis()

    def to_MM_SS(self, value):
        return f"{int(value / 60):02d}:{int(value % 60):02d}"

    def compute_graph_axis(self) -> None:
        logger.info("Compute graph x-axis")
        ds_factor = 1 if not self.get("vis_downsample") else DOWNSAMPLE_FACTOR

        if self.x_lim_max_old != self.x_lim_max:
            x_range = self.graph.getViewBox().viewRange()[0]
            x_min = (x_range[0] / self.x_lim_max_old) * self.x_lim_max
            x_max = (x_range[1] / self.x_lim_max_old) * self.x_lim_max
            self.graph.setLimits(xMin=0, xMax=self.x_lim_max)
            self.graph.setLimits(yMin=0, yMax=1)
            self.graph.setXRange(x_min, x_max)
        else:
            self.graph.setLimits(xMin=0, xMax=self.x_lim_max)
            self.graph.setLimits(yMin=0, yMax=1)

        self.graph.getAxis("left").setLabel("EAR Score")
        x_axis = self.graph.getAxis("bottom")

        if self.get("as_time"):
            x_axis.setLabel("Time (MM:SS)")
            try:
                # this exception occurs when the user enters nothing or a non-number
                fps = int(int(self.config.get("fps")) / ds_factor)
            except ValueError:
                fps = int(30 / ds_factor)

            x_ticks = np.arange(0, self.x_lim_max, fps)
            x_ticks_lab = [str(self.to_MM_SS(x // fps)) for x in x_ticks]

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
                    [(x, xl) for x, xl in zip(x_ticks[::10], x_ticks_lab[:10])],
                ]
            )

    def _analyse(self) -> None:
        self.progress.setRange(0, 100)
        if self.file is None:
            return
        self.progress.setValue(0)

        ear_l = self.raw_ear_l
        ear_r = self.raw_ear_r

        kwargs = {}
        kwargs["threshold_l"] = float(self.get("threshold_l"))
        kwargs["threshold_r"] = float(self.get("threshold_r"))
        kwargs["fps"] = int(self.get("fps"))
        kwargs["distance"] = float(self.get("min_dist"))
        kwargs["prominence"] = float(self.get("min_prominence"))
        kwargs["width_min"] = int(self.get("min_width"))
        kwargs["width_max"] = int(self.get("max_width"))

        self.progress.setValue(10)

        kwargs["smooth"] = self.get("smooth")
        smooth_size = int(self.get("smooth_size"))
        kwargs["smooth_size"] = smooth_size if smooth_size % 2 == 1 else (smooth_size + 1)
        kwargs["smooth_poly"] = int(self.get("smooth_poly"))

        self.ear_l = ear_l if not kwargs["smooth"] else blinking.smooth(ear_l, **kwargs)
        self.ear_r = ear_r if not kwargs["smooth"] else blinking.smooth(ear_r, **kwargs)

        self.progress.setValue(40)

        self.blinking_l = blinking.peaks(self.ear_l, threshold=kwargs["threshold_l"], **kwargs)
        self.progress.setValue(50)
        self.blinking_r = blinking.peaks(self.ear_r, threshold=kwargs["threshold_r"], **kwargs)
        self.progress.setValue(60)

        self.plot_data(self.ear_l, self.ear_r)

        self.progress.setValue(80)

        self.print_results(self.blinking_l, self.blinking_r, **kwargs)
        self.progress.setValue(100)

    def sec_to_min(self, seconds: float) -> str:
        minutes = int(seconds / 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

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
        blinking_l["time30"] = blinking_l["time30"].apply(self.sec_to_min)
        blinking_r["time30"] = blinking_r["time30"].apply(self.sec_to_min)
        blinking_l["timeFPS"] = blinking_l["timeFPS"].apply(self.sec_to_min)
        blinking_r["timeFPS"] = blinking_r["timeFPS"].apply(self.sec_to_min)

        self._reset_result_text()

        self._add("===Video Info===")
        self._add(f"File: {self.file.as_posix()}")
        self._add(f"Runtime: {self.sec_to_min(len(self.ear_l) / kwargs['fps'])}")

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

    def save_results(self) -> None:
        if self.file is None:
            return

        file_info = self.file.parent / (self.file.stem + "_blinking_info.txt")
        logger.info("Saving blinking results", file=file_info)
        file_info.write_text(self.result_text)
        self.blinking_l.to_excel(self.file.parent / (self.file.stem + "_blinking_l.xlsx"), sheet_name="blinking_l", index=False)
        self.blinking_r.to_excel(self.file.parent / (self.file.stem + "_blinking_r.xlsx"), sheet_name="blinking_r", index=False)
        logger.info("Saving blinking finished")

    def fill_tables(self, blinking_l: pd.DataFrame, blinking_r: pd.DataFrame) -> None:
        self.model_l.clear()
        self.model_r.clear()
        for _, row in blinking_l.iterrows():
            self.model_l.appendRow(self.to_qt_row(row))

        for _, row in blinking_r.iterrows():
            self.model_r.appendRow(self.to_qt_row(row))

        self.table_l.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table_r.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)

        self.model_l.setHorizontalHeaderLabels(list(blinking_l.columns))
        self.model_r.setHorizontalHeaderLabels(list(blinking_r.columns))

    def to_qt_row(self, row: pd.Series) -> list:
        return [QtGui.QStandardItem(str(row[c])) for c in row.index]

    def _show_peaks(self, plot: pg.ScatterPlotItem, blink: pd.DataFrame, settings: dict) -> None:
        if blink is None:
            return

        f = 1 if not self.get("vis_downsample") else DOWNSAMPLE_FACTOR

        peaks = blink["frame"].values / f
        score = blink["score"].values

        pen = pg.mkPen(settings)
        plot.setData(x=peaks, y=score, pen=pen)

        if not self.get("draw_width_height"):
            return

        val = self.progress.value()

        self.progress.setValue(0)
        for i, row in blink.iterrows():
            lh = self.graph.plot(
                [row["ips_l"] / f, row["ips_r"] / f],
                [row["height"], row["height"]],
                pen=pen,
            )
            lv = self.graph.plot(
                [row["frame"] / f, row["frame"] / f],
                [row["score"], row["score"] + row["promi"]],
                pen=pen,
            )
            self.lines.append(lh)
            self.lines.append(lv)
            self.progress.setValue(i / len(blink))

        self.progress.setValue(val)

    def _reset_result_text(self) -> None:
        self.result_text = ""
        self.te_results_g.setText("")

    def _add(self, text: str = "") -> None:
        self.result_text += text + "\n"

    def _set_result_text(self) -> None:
        self.te_results_g.setText(self.result_text)

    def _load_csv(self) -> None:
        logger.info("Open file dialo for loading CSV file")
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select csv file",
            ".",
            "Video Files (*.csv)",
            options=QtWidgets.QFileDialog.DontUseNativeDialog,
        )

        if fileName != "":
            logger.info("Try to load CSV file", file=fileName)
            self.file = pathlib.Path(fileName)
            self._load_file(self.file)
            logger.info("Loaded CSV file", file=fileName)
        else:
            logger.info("No file selected")

    def _load_file(self, path: pathlib.Path) -> None:
        self.progress.setValue(0)
        self.model_l.clear()
        self.model_r.clear()
        self.te_results_g.setText("")

        self.blinking_l = None
        self.blinking_r = None

        self.ear_l = None
        self.ear_r = None

        self.raw_ear_r, self.raw_ear_l = self.__handle_legacy_files(path)
        if self.raw_ear_r is None or self.raw_ear_l is None:
            return

        self.progress.setValue(40)
        self.plot_data()
        self.progress.setValue(100)

    def __handle_legacy_files(self, file: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
        logger.info("Check if file is legacy format", file=file.as_posix())

        df = pd.read_csv(file.as_posix(), sep=";")
        col_length = len(df.columns)
        if col_length == 1:
            logger.info("File is new format", file=file.as_posix())
            # this means has a different separator like ","
            # this should be the new format
            df = pd.read_csv(file.as_posix(), sep=",")
            if "dlib_ear_r" in df.columns:
                return df["dlib_ear_r"].values, df["dlib_ear_l"].values
            if "mediapipe_ear_r" in df.columns:
                return df["mediapipe_ear_r"].values, df["mediapipe_ear_l"].values

            logger.error("File does not contain any supported columns", file=file.as_posix())
            return None, None

        # if this case is reached, we know that the back was dlib so the renaming
        # with the prefix dlib is no problem at all
        if col_length > 1:
            cols = list(df.columns)
            if "ear_score_right" in cols:
                logger.info("File is legacy format [old]", file=file.as_posix())
                df = df.rename(
                    columns={
                        "ear_score_right": "dlib_ear_r",
                        "ear_score_left": "dlib_ear_l",
                        "valid": "dlib_ear_valid",
                    }
                )
                df.to_csv(file.as_posix(), sep=",", index=False)
                return df["dlib_ear_r"].values, df["dlib_ear_l"].values
            if "ear_score_rigth" in cols:
                logger.info("File is legacy format [spell]", file=file.as_posix())
                df = df.rename(
                    columns={
                        "ear_score_rigth": "dlib_ear_r",
                        "ear_score_left": "dlib_ear_l",
                        "valid": "valid_l",
                    }
                )
                df.to_csv(file.as_posix(), sep=",", index=False)
                return df["dlib_ear_r"].values, df["dlib_ear_l"].values

            else:
                logger.error(
                    "File has not supported content",
                    file=file.as_posix(),
                    columns=cols,
                )
                logger.error("Please contact the developer")
                return None, None

        self.file = file
        self._load_file(file)

    def clear(self) -> None:
        self.plot_peaks_l.clear()
        self.plot_peaks_r.clear()
        for line in self.lines:
            line.clear()

    def plot_data(self, data_l: np.ndarray = None, data_r: np.ndarray = None) -> None:
        self.clear()

        data_l = data_l if data_l is not None else self.ear_l
        data_r = data_r if data_r is not None else self.ear_r
        data_l = data_l if data_l is not None else self.raw_ear_l
        data_r = data_r if data_r is not None else self.raw_ear_r

        if data_l is None or data_r is None:
            return

        if self.get("vis_downsample"):
            data_l = signal.resample(data_l, int(len(data_l) / 8))
            data_r = signal.resample(data_r, int(len(data_r) / 8))
        self.x_lim_max_old = self.x_lim_max
        self.x_lim_max = len(data_l)

        self.plot_ear_l.setDownsampling(method="mean", auto=True)
        self.plot_ear_r.setDownsampling(method="mean", auto=True)

        self.plot_ear_l.setData(data_l)
        self.plot_ear_r.setData(data_r)

        self.compute_graph_axis()

        self._show_peaks(self.plot_peaks_l, self.blinking_l, {"color": "#00F", "width": 2})
        self._show_peaks(self.plot_peaks_r, self.blinking_r, {"color": "#F00", "width": 2})

    def shut_down(self) -> None:
        # this widget doesn't have any shut down requirements
        return
