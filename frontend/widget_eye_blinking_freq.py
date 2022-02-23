import json
import pathlib
from typing import Tuple

import numpy as np
import pandas as pd
import pyqtgraph as pg
import structlog
from pyqtconfig import ConfigManager
from qtpy import QtCore, QtGui, QtWidgets
from tabulate import tabulate

from jefapato import plotting
from jefapato.methods import blinking

DEFAULTS = {
    "smooth": True,
    "smooth_size": "91",
    "smooth_poly": "5",
    "min_dist": "80",
    "min_height": "0.1",
    "min_prominence": "0.05",
    "fps": "240",
    "min_width": "10",
    "max_width": "150",
    "threshold_l": "0.4",
    "threshold_r": "0.4",
    "draw_width_height": False,
}

logger = structlog.get_logger()
CONFIG_PATH = pathlib.Path("config/config_eye_blinking_freq.json")


class WidgetEyeBlinkingFreq(QtWidgets.QSplitter):
    def __init__(self):
        super().__init__()
        self.setOrientation(QtCore.Qt.Vertical)

        logger.info("Initializing EyeBlinkingFreq widget")

        try:
            with open(CONFIG_PATH, "r") as f:
                config = json.load(f)
                logger.info("Loaded config file for EyeBlinkingFreq widget")
                # if we add a new comfig item we need to add in case it is not
                # in the config file
                if len(config) < len(DEFAULTS):
                    logger.info("Config file is missing some items, adding them")
                    c = set(config.keys())
                    d = set(DEFAULTS.keys())
                    i = c.intersection(d)
                    for k in i:
                        config[k] = DEFAULTS[k]
        except FileNotFoundError:
            logger.info("Config file not found, using defaults")
            with open(CONFIG_PATH, "w") as f:
                json.dump(DEFAULTS, f)
                config = DEFAULTS

        self.result_text: str = ""
        self.config = ConfigManager(config)

        self.top_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, parent=self)
        self.graph_layout = pg.GraphicsLayoutWidget(parent=self)
        self.graph = plotting.WidgetGraph()
        self.graph.getViewBox().enableAutoRange(enable=False)
        self.graph.setYRange(0, 1)
        self.graph_layout.addItem(self.graph)

        self.model_l = QtGui.QStandardItemModel(self)
        self.model_r = QtGui.QStandardItemModel(self)

        self.table_l = QtWidgets.QTableView()
        self.table_l.setModel(self.model_l)

        self.table_r = QtWidgets.QTableView()
        self.table_r.setModel(self.model_r)

        self.settings = QtWidgets.QFormLayout()

        self.q = QtWidgets.QWidget()
        self.q.setLayout(self.settings)
        self.q.setMaximumHeight(600)

        t_l = QtWidgets.QVBoxLayout()
        w_t_l = QtWidgets.QWidget()
        w_t_l.setLayout(t_l)

        t_l.addWidget(QtWidgets.QLabel("Left Eye:"))
        t_l.addWidget(self.table_l)

        t_r = QtWidgets.QVBoxLayout()
        w_t_r = QtWidgets.QWidget()
        w_t_r.setLayout(t_r)

        t_r.addWidget(QtWidgets.QLabel("Right Eye:"))
        t_r.addWidget(self.table_r)

        self.top_splitter.addWidget(w_t_l)
        self.top_splitter.addWidget(w_t_r)
        self.top_splitter.addWidget(self.q)

        self.top_splitter.setStretchFactor(0, 35)
        self.top_splitter.setStretchFactor(1, 35)
        self.top_splitter.setStretchFactor(2, 30)

        self.button_load = QtWidgets.QPushButton("Load CSV File")
        self.button_load.clicked.connect(self._load_csv)
        self.button_anal = QtWidgets.QPushButton("Analyse")
        self.button_anal.clicked.connect(self._analyse)

        self.le_th_l = QtWidgets.QLineEdit()
        self.config.add_handler("threshold_l", self.le_th_l)
        self.le_th_l.setToolTip("Theshold for left eye")
        self.le_th_l.textChanged.connect(self._save_settings)

        self.le_th_r = QtWidgets.QLineEdit()
        self.config.add_handler("threshold_r", self.le_th_r)
        self.le_th_r.setToolTip("Theshold for right eye")
        self.le_th_r.textChanged.connect(self._save_settings)

        # settings
        self.le_fps = QtWidgets.QLineEdit()
        self.config.add_handler("fps", self.le_fps)
        self.le_fps.setValidator(QtGui.QIntValidator())
        self.le_fps.setToolTip("This is the number of frames per second.")
        self.le_fps.textChanged.connect(self._save_settings)

        self.le_distance = QtWidgets.QLineEdit()
        self.config.add_handler("min_dist", self.le_distance)
        self.le_distance.setToolTip(
            "This value controls the minimum distance between two blinks."
        )
        self.le_distance.textChanged.connect(self._save_settings)

        self.le_prominence = QtWidgets.QLineEdit()
        self.config.add_handler("min_prominence", self.le_prominence)
        self.le_prominence.setToolTip(
            "This value controls the minimum prominence of a blink."
        )
        self.le_prominence.textChanged.connect(self._save_settings)

        self.le_width_min = QtWidgets.QLineEdit()
        self.config.add_handler("min_width", self.le_width_min)
        self.le_width_min.setToolTip(
            "This value controls the minimum width of a blink."
        )
        self.le_width_min.setValidator(QtGui.QIntValidator())
        self.le_width_min.textChanged.connect(self._save_settings)

        self.le_width_max = QtWidgets.QLineEdit("150")
        self.config.add_handler("max_width", self.le_width_max)
        self.le_width_max.setToolTip(
            "This value controls the maximum width of a blink."
        )
        self.le_width_max.setValidator(QtGui.QIntValidator())
        self.le_width_max.textChanged.connect(self._save_settings)

        self.smooth = QtWidgets.QCheckBox()
        self.smooth.toggled.connect(self._save_settings)
        self.config.add_handler("smooth", self.smooth)
        self.smooth.setToolTip("Smooth the data")

        self.le_smooth_size = QtWidgets.QLineEdit()
        self.config.add_handler("smooth_size", self.le_smooth_size)
        self.le_smooth_size.setEnabled(self.smooth.isChecked())
        self.le_smooth_size.setValidator(QtGui.QIntValidator())
        self.le_smooth_size.setToolTip(
            "This value controls the size of the smoothing window."
        )
        self.le_smooth_size.textChanged.connect(self._save_settings)

        self.le_smooth_poly = QtWidgets.QLineEdit()
        self.config.add_handler("smooth_poly", self.le_smooth_poly)
        self.le_smooth_poly.setEnabled(self.smooth.isChecked())
        self.le_smooth_poly.setValidator(QtGui.QIntValidator())
        self.le_smooth_poly.setToolTip(
            "This value controls the polynomial order of the smoothing."
        )
        self.le_smooth_poly.textChanged.connect(self._save_settings)

        self.draw_width_height = QtWidgets.QCheckBox()
        self.draw_width_height.toggled.connect(self._save_settings)
        self.config.add_handler("draw_width_height", self.draw_width_height)

        self.smooth.toggled.connect(lambda value: self.le_smooth_size.setEnabled(value))
        self.smooth.toggled.connect(lambda value: self.le_smooth_poly.setEnabled(value))

        self.te_results_g = QtWidgets.QTextEdit()
        self.te_results_g.setFontFamily("mono")
        self.te_results_g.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.NoWrap)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)

        self.settings.addRow(self.button_load)
        self.settings.addRow("Threshold Left:", self.le_th_l)
        self.settings.addRow("Threshold Right", self.le_th_r)
        self.settings.addRow("FPS:", self.le_fps)
        self.settings.addRow("Min. Distance:", self.le_distance)
        self.settings.addRow("Min. Prominence:", self.le_prominence)
        self.settings.addRow("Min. Peak Width:", self.le_width_min)
        self.settings.addRow("Max. Peak Width:", self.le_width_max)

        self.settings.addRow("Smooth:", self.smooth)
        self.settings.addRow("Smooth Window:", self.le_smooth_size)
        self.settings.addRow("Smooth Polynom:", self.le_smooth_poly)

        self.settings.addRow("Draw Width/Height:", self.draw_width_height)

        self.settings.addRow(self.button_anal)
        self.settings.addRow(self.te_results_g)
        self.settings.addRow(self.progress)

        self.ear_l = np.zeros(1000, dtype=np.float32)
        self.ear_r = np.zeros(1000, dtype=np.float32)

        self.plot_ear_l = self.graph.add_curve({"color": "#00F", "width": 2})
        self.plot_ear_r = self.graph.add_curve({"color": "#F00", "width": 2})

        self.plot_peaks_l = self.graph.add_scatter()
        self.plot_peaks_r = self.graph.add_scatter()

        self.lines = list()
        self.file = None

        logger.info("Initialized EyeBlinkingFreq widget")

    def _save_settings(self):
        with (open(CONFIG_PATH, "w")) as f:
            json.dump(self.config.as_dict(), f)

    def _analyse(self) -> None:
        if self.file is None:
            return
        self.progress.setValue(0)

        ear_l = self.ear_l
        ear_r = self.ear_r

        kwargs = {}
        kwargs["threshold_l"] = float(self.config.get("threshold_l"))
        kwargs["threshold_r"] = float(self.config.get("threshold_r"))
        kwargs["fps"] = int(self.config.get("fps"))
        kwargs["distance"] = float(self.config.get("min_dist"))
        kwargs["prominence"] = float(self.config.get("min_prominence"))
        kwargs["width_min"] = int(self.config.get("min_width"))
        kwargs["width_max"] = int(self.config.get("max_width"))

        self.progress.setValue(10)

        kwargs["smooth"] = self.config.get("smooth")
        smooth_size = int(self.config.get("smooth_size"))
        kwargs["smooth_size"] = (
            smooth_size if smooth_size % 2 == 1 else (smooth_size + 1)
        )
        kwargs["smooth_poly"] = int(self.config.get("smooth_poly"))

        ear_l = ear_l if not kwargs["smooth"] else blinking.smooth(ear_l, **kwargs)
        ear_r = ear_r if not kwargs["smooth"] else blinking.smooth(ear_r, **kwargs)

        self.progress.setValue(30)
        self._set_data(ear_l, ear_r)
        self.progress.setValue(40)

        blinking_l = blinking.peaks(ear_l, threshold=kwargs["threshold_l"], **kwargs)
        self.progress.setValue(50)
        blinking_r = blinking.peaks(ear_r, threshold=kwargs["threshold_r"], **kwargs)
        self.progress.setValue(60)

        self.plot_results(blinking_l, blinking_r)
        self.progress.setValue(80)

        self.print_results(blinking_l, blinking_r, **kwargs)
        self.progress.setValue(100)

    def plot_results(
        self,
        blinking_l: pd.DataFrame,
        blinking_r: pd.DataFrame,
    ):
        for ll in self.lines:
            ll.clear()
        self.lines.clear()

        self._show_peaks(self.plot_peaks_l, blinking_l, {"color": "#00F", "width": 2})
        self._show_peaks(self.plot_peaks_r, blinking_r, {"color": "#F00", "width": 2})

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
            df = blinking_l[
                (blinking_l["frame"] >= start) & (blinking_l["frame"] < stop)
            ]
            _mean = np.mean(df["width"])
            _std = np.std(df["width"])
            self._add(f"Minute {index:02d} L: {_mean: 6.3f} +/- {_std: 6.3f}[frames]")

            _mean /= kwargs["fps"]
            _std /= kwargs["fps"]
            self._add(f"Minute {index:02d} R: {_mean: 6.3f} +/- {_std: 6.3f}[s]")

        self._add()

        for index, (start, stop) in enumerate(zip(bins[:-2], bins[1:])):
            df = blinking_r[
                (blinking_r["frame"] >= start) & (blinking_r["frame"] < stop)
            ]
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

        self.save_results(blinking_l, blinking_r)

    def save_results(self, blinking_l: pd.DataFrame, blinking_r: pd.DataFrame) -> None:
        file_info = self.file.parent / (self.file.stem + "_blinking_info.txt")

        with open(file_info, "w") as f:
            f.write(self.result_text)

        file_blinking_l = self.file.parent / (self.file.stem + "_blinking_l.csv")
        file_blinking_r = self.file.parent / (self.file.stem + "_blinking_r.csv")

        blinking_l.to_csv(file_blinking_l)
        blinking_r.to_csv(file_blinking_r)

    def fill_tables(self, blinking_l: pd.DataFrame, blinking_r: pd.DataFrame) -> None:
        self.model_l.clear()
        self.model_r.clear()
        for _, row in blinking_l.iterrows():
            self.model_l.appendRow(self.to_qt_row(row))

        for _, row in blinking_r.iterrows():
            self.model_r.appendRow(self.to_qt_row(row))

        self.table_l.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.table_r.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )

        self.model_l.setHorizontalHeaderLabels(list(blinking_l.columns))
        self.model_r.setHorizontalHeaderLabels(list(blinking_r.columns))

    def to_qt_row(self, row: pd.Series) -> list:
        return [QtGui.QStandardItem(str(row[c])) for c in row.index]

    def _show_peaks(
        self, plot: pg.ScatterPlotItem, blink: pd.DataFrame, settings: dict
    ) -> None:
        peaks = blink["frame"]
        score = blink["score"]

        pen = pg.mkPen(settings)
        plot.setData(x=peaks, y=score, pen=pen)

        if not self.config.get("draw_width_height"):
            return

        for _, row in blink.iterrows():
            lh = self.graph.plot(
                [row["ips_l"], row["ips_r"]], [row["height"], row["height"]], pen=pen
            )
            lv = self.graph.plot(
                [row["frame"], row["frame"]],
                [row["score"], row["score"] + row["promi"]],
                pen=pen,
            )
            self.lines.append(lh)
            self.lines.append(lv)

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
        self.ear_r, self.ear_l = self.__handle_legacy_files(path)
        if self.ear_r is None or self.ear_l is None:
            return

        self.progress.setValue(40)

        self._set_data(self.ear_l, self.ear_r)
        self.model_l.clear()
        self.model_r.clear()
        for line in self.lines:
            line.clear()
        self.progress.setValue(70)

        self.plot_peaks_l.clear()
        self.plot_peaks_r.clear()

        self.te_results_g.setText("")
        self.progress.setValue(100)

    def __handle_legacy_files(
        self, file: pathlib.Path
    ) -> Tuple[np.ndarray, np.ndarray]:
        logger.info("Check if file is legacy format", file=file.as_posix())

        df = pd.read_csv(file.as_posix(), sep=";")
        col_length = len(df.columns)
        if col_length == 1:
            logger.info("File is new format", file=file.as_posix())
            # this means has a different separator like ","
            # this should be the new format
            df = pd.read_csv(file.as_posix(), sep=",")
            if "dlib_ear_r" in df.columns:
                return df["dlib_ear_r"], df["dlib_ear_l"]

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
                return df["dlib_ear_r"], df["dlib_ear_l"]
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
                return df["dlib_ear_r"], df["dlib_ear_l"]

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

    def _set_data(self, data_l: np.ndarray, data_r: np.ndarray) -> None:
        self.plot_ear_l.setDownsampling(method="mean", auto=True)
        self.plot_ear_r.setDownsampling(method="mean", auto=True)

        self.plot_ear_l.setData(data_l)
        self.plot_ear_r.setData(data_r)

        self.graph.setLimits(xMin=0, xMax=len(data_l))
        self.graph.setXRange(0, len(self.ear_l))
