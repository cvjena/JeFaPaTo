import csv
from pathlib import Path
from typing import List

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTableView,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from scipy.signal import find_peaks, savgol_filter

from jefapato.plotter import GraphWidget


class WidgetEyeBlinkingFreq(QSplitter):
    def __init__(self):
        super().__init__()
        self.setOrientation(Qt.Vertical)

        self.top_splitter = QSplitter(Qt.Horizontal, parent=self)
        self.graph = GraphWidget(self)

        self.model_l = QtGui.QStandardItemModel(self)
        self.model_r = QtGui.QStandardItemModel(self)

        self.table_l = QTableView()
        self.table_l.setModel(self.model_l)

        self.table_r = QTableView()
        self.table_r.setModel(self.model_r)

        self.settings = QFormLayout()

        self.q = QWidget()
        self.q.setLayout(self.settings)
        self.q.setMaximumHeight(600)

        t_l = QVBoxLayout()
        w_t_l = QWidget()
        w_t_l.setLayout(t_l)

        t_l.addWidget(QLabel("Left Eye:"))
        t_l.addWidget(self.table_l)

        t_r = QVBoxLayout()
        w_t_r = QWidget()
        w_t_r.setLayout(t_r)

        t_r.addWidget(QLabel("Right Eye:"))
        t_r.addWidget(self.table_r)

        self.top_splitter.addWidget(self.q)
        self.top_splitter.addWidget(w_t_l)
        self.top_splitter.addWidget(w_t_r)

        self.top_splitter.setStretchFactor(0, 3)
        self.top_splitter.setStretchFactor(1, 3)
        self.top_splitter.setStretchFactor(2, 3)

        self.button_load = QPushButton("Load CSV File")
        self.button_load.clicked.connect(self._load)
        self.button_anal = QPushButton("Analyse")
        self.button_anal.clicked.connect(self._analyse)

        self.smooth = QCheckBox()
        self.smooth.setChecked(True)

        self.le_th_l = QLineEdit("0.2")
        self.le_th_r = QLineEdit("0.2")

        self.graph.signal_y_ruler_changed.connect(
            lambda value: self.le_th_l.setText(f"{value:05.3f}")
        )
        self.graph.signal_y_ruler_changed.connect(
            lambda value: self.le_th_r.setText(f"{value:05.3f}")
        )

        self.le_fps = QLineEdit("240")
        self.le_fps.setValidator(QtGui.QIntValidator())

        self.le_smooth_size = QLineEdit("51")
        self.le_smooth_size.setEnabled(self.smooth.isChecked())
        self.le_smooth_size.setValidator(QtGui.QIntValidator())

        self.le_smooth_poly = QLineEdit("3")
        self.le_smooth_poly.setEnabled(self.smooth.isChecked())
        self.le_smooth_poly.setValidator(QtGui.QIntValidator())

        self.smooth.toggled.connect(lambda value: self.le_smooth_size.setEnabled(value))
        self.smooth.toggled.connect(lambda value: self.le_smooth_poly.setEnabled(value))

        self.te_results_g = QTextEdit()
        self.te_results_g.setFontFamily("mono")

        self.settings.addRow(self.button_load)
        self.settings.addRow("Threshold Left:", self.le_th_l)
        self.settings.addRow("Threshhold Rirgt", self.le_th_r)
        self.settings.addRow("FPS:", self.le_fps)
        self.settings.addRow("Smooth:", self.smooth)
        self.settings.addRow("Smooth Window:", self.le_smooth_size)
        self.settings.addRow("Smooth Polynom:", self.le_smooth_poly)
        self.settings.addRow(self.button_anal)
        self.settings.addRow(self.te_results_g)

        self.ear_l: list[float] = list()
        self.ear_r: list[float] = list()

        self.plot_ear_l = self.graph.add_curve({"color": "#00F", "width": 2})
        self.plot_ear_r = self.graph.add_curve({"color": "#F00", "width": 2})

        self.plot_peaks_l = self.graph.add_scatter()
        self.plot_peaks_r = self.graph.add_scatter()

        self.file = Path("/home/tim/data/JeFaPaTo/Proband 06_2021-09-24_14-56-34.csv")
        self._load_file(self.file)

    def _analyse(self) -> None:
        th_l = float(self.le_th_l.text())
        th_r = float(self.le_th_r.text())

        fps = int(self.le_fps.text())

        val_l = np.array(self.ear_l)
        val_r = np.array(self.ear_r)

        smooth = self.smooth.isChecked()
        w_size = int(self.le_smooth_size.text())
        w_size = w_size if w_size % 2 == 1 else (w_size + 1)
        polynom = int(self.le_smooth_poly.text())

        if smooth:
            val_l = savgol_filter(val_l, w_size, polyorder=polynom)
            val_r = savgol_filter(val_r, w_size, polyorder=polynom)

        self._set_data(val_l.tolist(), val_r.tolist())

        peaks_l, _ = find_peaks(-val_l, distance=150)
        peaks_r, _ = find_peaks(-val_r, distance=150)

        peaks_l = peaks_l[val_l[peaks_l] < th_l]
        peaks_r = peaks_r[val_r[peaks_r] < th_r]

        self._show_peaks(self.plot_peaks_l, val_l, peaks_l)
        self._show_peaks(self.plot_peaks_r, val_r, peaks_r)

        result = ""

        total_seconds = len(val_l) / fps
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60

        bins = list()
        # plus 2 so we have all minutes and the remaining seconds
        for i in range(minutes + 2):
            bins.append(i * 60 * fps)

        hist_l, _ = np.histogram(peaks_l, bins=bins)
        hist_r, _ = np.histogram(peaks_r, bins=bins)

        result = "===Video Info===\n"
        result += f"File: {self.file.as_posix()}\n"
        result += f"Runtime: {minutes}m {seconds:.2f}s [total: {total_seconds:.2f}s]\n"
        result += f"FPS: {fps}\n"
        result += f"Threshold Left: {th_l}\n"
        result += f"Threshold Right: {th_r}\n"

        parameter = f"[Window Size: {w_size}, Polynomial: {polynom}]" if smooth else ""
        result += f"Smooth: {smooth} {parameter}\n"

        result += "\n"

        result += "===Blinking Info===\n"
        result += f"Blinks Per Mintue Left: {hist_l.tolist()}\n"
        result += f"Blinks Per Mintue Right: {hist_r.tolist()}\n"

        result += f"Average Left: {np.mean(hist_l): 6.3f}\n"
        result += f"Average Right: {np.mean(hist_r): 6.3f}\n"

        result += f"Average[wo/ last minute] Left: {np.mean(hist_l[:-1]): 6.3f}\n"
        result += f"Average[wo/ last minute] Right: {np.mean(hist_r[:-1]): 6.3f}\n"

        self.model_l.clear()
        self.model_r.clear()

        result += "\n"

        result += "===Detail Left Info===\n"
        for i, p in enumerate(peaks_l):
            items = [
                QtGui.QStandardItem(str(p)),
                QtGui.QStandardItem(str(val_l[p])),
            ]
            self.model_l.appendRow(items)
            result += f"{i+1:03d}; Frame {p: 7d}; EAR_SCORE {val_l[p]:05.3f}\n"

        result += "\n"
        result += "===Detail Right Info===\n"
        for i, p in enumerate(peaks_r):
            items = [
                QtGui.QStandardItem(str(p)),
                QtGui.QStandardItem(str(val_r[p])),
            ]
            self.model_r.appendRow(items)
            result += f"{i+1:03d}; Frame {p: 7d}; EAR_SCORE {val_r[p]:05.3f}\n"

        self.te_results_g.setText(result)

        self.table_l.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table_r.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        self.model_l.setHorizontalHeaderLabels(["Frame", "EAR_SCORE"])
        self.model_r.setHorizontalHeaderLabels(["Frame", "EAR_SCORE"])

    def _show_peaks(
        self, plot: pg.ScatterPlotItem, data: np.ndarray, peaks: np.ndarray
    ) -> None:
        plot.setData(x=peaks.tolist(), y=data[peaks].tolist())

    def _load(self) -> None:
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "Select csv file",
            ".",
            "Video Files (*.csv)",
        )

        if fileName != "":
            self.file = Path(fileName)
            self._load_file(self.file)

    def _load_file(self, path: Path) -> None:
        self.ear_l.clear()
        self.ear_r.clear()

        with open(path) as file:

            reader = csv.reader(file, delimiter=";")
            next(reader)
            for row in reader:
                self.ear_l.append(float(row[2]))
                self.ear_r.append(float(row[3]))

            self._set_data(self.ear_l, self.ear_r)
            self.model_l.clear()
            self.model_r.clear()
            self.te_results_g.setText("")

    def _set_data(self, data_l: List[float], data_r: List[float]) -> None:
        fps = int(self.le_fps.text())

        self.plot_ear_l.setData(data_l)
        self.plot_ear_r.setData(data_r)

        self.graph.setLimits(xMin=0, xMax=len(data_l))
        self.graph.setXRange(0, len(self.ear_l))

        if len(data_l) > 10000:
            self.graph.remove_grid()
            self.graph.axis_b.setTickSpacing(fps * 200, fps)
