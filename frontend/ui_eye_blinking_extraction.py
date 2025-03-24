from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
import qtawesome as qta
import structlog
from PyQt6.QtCore import pyqtSignal
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtWidgets import QMessageBox

from frontend import config, jwidgets
from jefapato import blinking

logger = structlog.get_logger()

#### Recommended Extracton Settings for Eye Blinking ####
# @30 FPS
# Minimum Distance: 10 Frames
# Minimum Prominence: 0.1 EAR Score
# Minimum Internal Width: 4 Frames
# Maximum Internal Width: 20 Frames
# Maximum Matching Distance: 15 Frames
# Partial Threshold Left:  0.18 EAR Score
# Partial Threshold Right: 0.18 EAR Score
#
# Smoothing
# - Window Size: 7
# - Polynomial Degree: 3
#
# ---
#
# @240 FPS
# Minimum Distance: 50 Frames
# Minimum Prominence: 0.1 EAR Score
# Minimum Internal Width: 20 Frames
# Maximum Internal Width: 100 Frames
# Maximum Matching Distance: 30 Frames
# Partial Threshold Left:  0.18 EAR Score
# Partial Threshold Right: 0.18 EAR Score
#
# Smoothing
# - Window Size: 7
# - Polynomial Degree: 3


DOWNSAMPLE_FACTOR = 8


def to_MM_SS(value):
    """
    Converts a value in seconds to a string representation in the format MM:SS.

    Args:
        value (int): The value in seconds to be converted.

    Returns:
        str: The string representation of the value in the format MM:SS.
    """
    return f"{int(value / 60):02d}:{int(value % 60):02d}"


def sec_to_min(seconds: float) -> str:
    """
    Converts seconds to minutes and seconds format.

    Args:
        seconds (float): The number of seconds to convert.

    Returns:
        str: The converted time in the format "MM:SS".
    """
    minutes = int(seconds / 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


# TODO just make this a normal widget and not a splitter
class EyeBlinkingExtraction(QtWidgets.QSplitter, config.Config):
    updated = pyqtSignal(int)

    def __init__(self, parent):
        config.Config.__init__(self, prefix="ear")
        QtWidgets.QSplitter.__init__(self, parent=parent)

        logger.info("Initializing EyeBlinkingFreq widget")
        self.result_text: str = ""

        self.x_lim_max = 1000
        self.raw_ear_l: np.ndarray | None = None
        self.raw_ear_r: np.ndarray | None = None
        self.ear_l: np.ndarray | None = None
        self.ear_r: np.ndarray | None = None

        self.blinking_l: pd.DataFrame | None = None
        self.blinking_r: pd.DataFrame | None = None
        self.blinking_matched: pd.DataFrame | None = None
        self.blinking_summary: pd.DataFrame | None = None

        self.lines: list = []
        self.file: Path | None = None
        self.data_frame: pd.DataFrame | None = None
        self.data_frame_columns: list[str] = []

        self.graph = jwidgets.JGraph(x_lim_max=self.x_lim_max)
        self.curve_l = self.graph.add_curve({"color": "#00F", "width": 2})  # TODO add correct colors...
        self.curve_r = self.graph.add_curve({"color": "#F00", "width": 2})  # TODO add correct colors...
        self.scatter_l_comp = self.graph.add_scatter()
        self.scatter_r_comp = self.graph.add_scatter()
        self.scatter_l_part = self.graph.add_scatter()
        self.scatter_r_part = self.graph.add_scatter()

        # UI elements
        self.setOrientation(QtCore.Qt.Orientation.Horizontal)

        self.setAcceptDrops(True)
        # Create the main layouts of the interface
        widget_content = QtWidgets.QWidget()
        self.layout_content = QtWidgets.QVBoxLayout()
        widget_content.setLayout(self.layout_content)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setFixedWidth(400)
        scroll_area.setMinimumHeight(200)
        scroll_area.setWidgetResizable(True)

        widget_settings = QtWidgets.QWidget()
        self.layout_settings = QtWidgets.QVBoxLayout()
        widget_settings.setLayout(self.layout_settings)

        scroll_area.setWidget(widget_settings)

        self.addWidget(widget_content)
        self.addWidget(scroll_area)

        self.setStretchFactor(0, 6)
        self.setStretchFactor(1, 4)

        # Create the specific widgets for the content layout
        self.tab_widget_results = QtWidgets.QTabWidget()

        # upper main content is a tab widget with the tables and text information
        # first tabe is the tables with the results

        self.table_matched = jwidgets.JTableBlinking()
        self.table_matched.selection_changed.connect(self.highlight_blink)

        # second tab is the text information
        self.table_summary = jwidgets.JTableSummary()

        self.graph_summary_visual = pg.GraphicsLayoutWidget()
        self.summary_visual = pg.ViewBox(invertY=True, lockAspect=True, enableMenu=True, enableMouse=True)
        self.summary_visual_image = pg.ImageItem()
        self.summary_visual.addItem(self.summary_visual_image)
        self.graph_summary_visual.addItem(self.summary_visual)

        self.tab_widget_results.addTab(self.table_matched, "Table Blinking")
        self.tab_widget_results.addTab(self.table_summary, "Table Summary")
        self.tab_widget_results.addTab(self.graph_summary_visual, "Visual Summary")

        # lower main content/ is a graph
        vb = self.graph.getViewBox()
        assert vb is not None, "Somehow the viewbox is None"
        vb.enableAutoRange(enable=False)
        self.graph.setYRange(0, 1)

        # Create the specific widgets for the settings layout
        self.layout_content.addWidget(self.tab_widget_results, stretch=1)
        self.layout_content.addWidget(self.graph, stretch=1)

        # Create the specific widgets for the settings layout, algorithm specific settings
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

        self.progress = self.parent().progress_bar  # type: ignore
        self.btn_load.clicked.connect(self.load_dialog)
        self.btn_anal.clicked.connect(self.extract_blinks)
        self.btn_summ.clicked.connect(self.compute_summary)
        self.btn_eprt.clicked.connect(self.save_results)

        # algorithm settings box
        self.box_settings = QtWidgets.QGroupBox("Blinking Extraction Settings")
        # dont make the groupbox changeable in height
        self.box_settings.setMinimumHeight(200)
        self.set_algo = QtWidgets.QGridLayout()

        self.jsmooth = jwidgets.JSmoothing(self)
        self.janalysis = jwidgets.JBlinkingAnalysis(self)
        self.janalysis.cb_video_fps.currentIndexChanged.connect(self.compute_graph_axis)
        self.jpeaks = jwidgets.JPeaks(self)
        self.jespbm = jwidgets.JESPBM(self)

        self.algorith_extract_tabs = QtWidgets.QTabWidget()
        self.algorith_extract_tabs.addTab(self.jpeaks, "Peaks")
        self.algorith_extract_tabs.addTab(self.jespbm, "ESPBM (Beta)")

        self.set_algo.addWidget(self.jsmooth, 0, 0, 1, 3)
        self.set_algo.addWidget(self.janalysis, 1, 0, 1, 3)
        self.set_algo.addWidget(self.algorith_extract_tabs, 2, 0, 1, 3)

        # Visual Settings #
        self.box_visuals = QtWidgets.QGroupBox("Graph Control")
        self.set_visuals = QtWidgets.QFormLayout()

        self.cb_as_time = QtWidgets.QCheckBox("Show Time")
        self.add_handler("as_time", self.cb_as_time)
        self.set_visuals.addRow(self.cb_as_time)
        self.cb_as_time.stateChanged.connect(self.compute_graph_axis)

        btn_reset_graph = QtWidgets.QPushButton(qta.icon("msc.refresh"), "Reset Graph Y Range")
        btn_reset_graph.clicked.connect(lambda: self.graph.setYRange(-0.5, 1))
        self.set_visuals.addRow(btn_reset_graph)

        btn_reset_view = QtWidgets.QPushButton(qta.icon("msc.refresh"), "View Full Graph")
        btn_reset_view.clicked.connect(lambda: self.graph.autoRange())
        self.set_visuals.addRow(btn_reset_view)

        # Export Settings #
        self.overwrite_export = QtWidgets.QCheckBox("Overwrite Existing File")
        self.add_handler("overwrite_export", self.overwrite_export, default=True)

        self.format_export = QtWidgets.QComboBox()
        self.format_export.addItems(["CSV", "Excel"])
        self.format_export.setCurrentIndex(0)
        self.add_handler("export_format", self.format_export, default="Excel")

        self.face_preview = jwidgets.JVideoFacePreview()
        self.face_preview.setMaximumHeight(300)

        # Layouting #
        self.box_settings.setLayout(self.set_algo)
        self.box_visuals.setLayout(self.set_visuals)

        # add all things to the settings layout
        self.layout_settings.addWidget(self.btn_load)
        self.layout_settings.addWidget(self.la_current_file)
        self.layout_settings.addWidget(self.face_preview, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.layout_settings.addWidget(jwidgets.JHLine())

        self.layout_settings.addWidget(QtWidgets.QLabel("Left Eye"))
        self.layout_settings.addWidget(self.comb_ear_l)
        self.layout_settings.addWidget(QtWidgets.QLabel("Right Eye"))
        self.layout_settings.addWidget(self.comb_ear_r)
        self.layout_settings.addWidget(jwidgets.JHLine())

        self.layout_settings.addWidget(self.box_settings)
        self.layout_settings.addWidget(self.btn_anal)
        self.layout_settings.addWidget(self.btn_summ)
        self.layout_settings.addWidget(jwidgets.JHLine())
        self.layout_settings.addWidget(QtWidgets.QLabel("Export Format"))
        self.layout_settings.addWidget(self.overwrite_export)
        self.layout_settings.addWidget(self.format_export)
        self.layout_settings.addWidget(self.btn_eprt)
        self.layout_settings.addWidget(jwidgets.JHLine())

        self.layout_settings.addWidget(self.box_visuals)
        self.layout_settings.addWidget(jwidgets.JHLine())

        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.layout_settings.addWidget(spacer)

        logger.info("Initialized EyeBlinkingFreq widget")
        self.compute_graph_axis()

        self.disable_column_selection()
        self.disable_algorithm()
        self.disable_export()

    def get_selected_fps(self) -> int:
        """
        Get the selected frames per second (fps) from the video fps combo box.

        Returns:
            int: The selected frames per second (fps).
        """
        return int(self.janalysis.cb_video_fps.currentText())

    # loading of the file
    def load_dialog(self) -> None:
        """
        Open a file dialog for loading a CSV file.
        """
        logger.info("Open file dialog for loading CSV file")
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select csv file",
            ".",
            "Video Files (*.csv)",
        )

        if file_path == "":
            logger.info("No file selected")
            return

        logger.info("File selected via dialog", file=file_path)
        self.load_file(Path(file_path))

    def load_file(self, file_path: Path) -> None:
        """
        Load a file and parse it as a data frame.

        Args:
            file_path (Path): The path to the file to be loaded.

        Returns:
            None
        """
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

        # compute the possible options for the columns
        # make all columns to lower case and find the one ending in ear_l and ear_r
        cols_lower = [x.lower() for x in self.data_frame_columns]
        idx_l = [i for i, x in enumerate(cols_lower) if x.endswith("_l")]
        if len(idx_l) > 0:
            self.comb_ear_l.setCurrentIndex(idx_l[0])
        idx_r = [i for i, x in enumerate(cols_lower) if x.endswith("_r")]
        if len(idx_r) > 0:
            self.comb_ear_r.setCurrentIndex(idx_r[0])

    def select_column_left(self, index: int) -> None:
        """
        Selects the left column of the data frame based on the given index.

        Args:
            index (int): The index of the column to select.

        Returns:
            None
        """
        if self.data_frame is None or self.data_frame_columns is None:
            return

        self.raw_ear_l = self.data_frame[self.data_frame_columns[index]].to_numpy()
        self.update_plot_raw()
        self.disable_export()

    def select_column_right(self, index: int) -> None:
        """
        Selects the column at the given index from the data frame and updates the plot.

        Args:
            index (int): The index of the column to select.

        Returns:
            None
        """
        if self.data_frame is None or self.data_frame_columns is None:
            return
        self.raw_ear_r = self.data_frame[self.data_frame_columns[index]].to_numpy()

        self.update_plot_raw()
        self.disable_export()

    def update_plot_raw(self) -> None:
        """
        Update the raw plot curves with new data.

        Clears the existing plot curves for left and right eye EAR values,
        and then sets the data for the plot curves based on the raw EAR values
        if they are available. Finally, it computes the graph axis.

        Returns:
            None
        """
        self.curve_l.clear()
        self.curve_r.clear()

        if self.raw_ear_l is not None:
            self.curve_l.setData(self.raw_ear_l)
        if self.raw_ear_r is not None:
            self.curve_r.setData(self.raw_ear_r)

        self.compute_graph_axis()

    def compute_graph_axis(self) -> None:
        """
        Compute the graph axis for the eye blinking extraction UI.

        This method sets the x-axis and y-axis limits for the graph, as well as the labels and ticks.
        If the 'as_time' flag is True, the x-axis is labeled as time in MM:SS format, otherwise it is labeled as frames.
        The y-axis is always labeled as "EAR Score".

        Returns:
            None
        """
        logger.info("Compute graph x-axis")
        ds_factor = 1 if not self.get("vis_downsample") else DOWNSAMPLE_FACTOR

        # compute the x_lim_max
        if self.raw_ear_l is not None and self.raw_ear_r is not None:
            self.x_lim_max = max(len(self.raw_ear_l), len(self.raw_ear_r))

        self.graph.setLimits(xMin=0, yMin=0, xMax=self.x_lim_max, yMax=1)
        self.graph.setXRange(0, self.x_lim_max)
        self.graph.setYRange(0, 1)

        self.graph.getAxis("left").setLabel("EAR Score")
        x_axis = self.graph.getAxis("bottom")

        if self.get("as_time"):
            x_axis.setLabel("Time (MM:SS)")
            fps = self.get_selected_fps()
            x_ticks = np.arange(0, self.x_lim_max, fps)
            x_ticks_lab = [str(to_MM_SS(x // fps)) for x in x_ticks]

            # TODO add some milliseconds to the ticks
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
        """
        Extracts eye blinks from the data and updates the UI accordingly.

        Returns:
            None
        """
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        if not self.validate_compute_parameters():
            return
        self.progress.setValue(30)

        if not self.compute_intervals():
            return
        self.progress.setValue(60)

        if not self.plot_intervals():
            return
        self.progress.setValue(90)

        if self.blinking_matched is None:
            return

        self.table_matched.set_data(self.blinking_matched)
        self.progress.setValue(100)

        self.enable_export()
        self.tab_widget_results.setCurrentIndex(0)

    def validate_compute_parameters(self) -> bool:
        """
        Validate the parameters for computing the intervals for eye blinking extraction.

        Returns:
            bool: True if the parameters are valid, False otherwise.
        """
        # check if the column selection index are not the same
        if self.comb_ear_l.currentIndex() == self.comb_ear_r.currentIndex():
            logger.error("The same column is selected for both eyes")
            QMessageBox.critical(
                self,
                title="Blinking Extraction Error",
                text="Both EAR columns are the same! Please select different columns and try again",
            )
            return False

        def validate_setting(setting_name: str) -> bool:
            try:
                _ = self.get(setting_name)
            except ValueError:
                logger.error("Error while validating the settings", setting=setting_name)
                QMessageBox.critical(None, "Blinking Extraction", f"The setting {setting_name} is not a valid input. Please change your settings and try again")
                return False
            return True

        if not validate_setting("partial_threshold_l"):
            return False
        if not validate_setting("partial_threshold_r"):
            return False
        if not validate_setting("min_dist"):
            return False
        if not validate_setting("min_prominence"):
            return False
        if not validate_setting("min_width"):
            return False
        if not validate_setting("maximum_matching_dist"):
            return False
        if not validate_setting("max_width"):
            return False
        if not validate_setting("smooth_size"):
            return False
        if not validate_setting("smooth_poly"):
            return False

        if (self.get("partial_threshold_l") == "auto" and self.get("partial_threshold_r") != "auto") or (self.get("partial_threshold_l") != "auto" and self.get("partial_threshold_r") == "auto"):
            QMessageBox.critical(None, "Blinking Extraction Warning", "Both partial thresholds need to be set to 'auto' or a value")
            return False

        return True

    def compute_intervals(self) -> bool:
        """
        Computes the intervals for eye blinking extraction based on the provided settings.

        Returns:
            bool: True if the computation is successful, False otherwise.
        """
        if self.raw_ear_l is None or self.raw_ear_r is None:
            return False

        smooth_size = self.geti("smooth_size", 7)
        smooth_poly = self.geti("smooth_poly", 3)

        try:
            self.ear_l = blinking.smooth(self.raw_ear_l, smooth_size=smooth_size, smooth_poly=smooth_poly) if self.getb("smooth") else self.raw_ear_l
            self.ear_r = blinking.smooth(self.raw_ear_r, smooth_size=smooth_size, smooth_poly=smooth_poly) if self.getb("smooth") else self.raw_ear_r
        except Exception as e:
            logger.error("Error while smoothing the EAR data", error=e)
            QMessageBox.critical(None, "Blinking Extraction Error", f"The EAR data could not be smoothed. {e}")
            return False
        self.progress.setValue(40)

        # if only one is set to auto, inform the user
        partial_threshold_l_value = self.get("partial_threshold_l")
        partial_threshold_r_value = self.get("partial_threshold_r")
        partial_threshold_l = "auto" if partial_threshold_l_value == "auto" else float(partial_threshold_l_value) if partial_threshold_l_value is not None else 0.0
        partial_threshold_r = "auto" if partial_threshold_r_value == "auto" else float(partial_threshold_r_value) if partial_threshold_r_value is not None else 0.0

        try:
            if self.algorith_extract_tabs.currentWidget() == self.jpeaks:
                min_dist = self.geti("min_dist", 50)
                min_prom = self.getf("min_prominence", 0.1)
                min_int_width = self.geti("min_width", 10)
                max_int_width = self.geti("max_width", 100)

                self.blinking_l, self.comp_partial_threshold_l = blinking.peaks(
                    time_series=self.ear_l,
                    minimum_distance=min_dist,
                    minimum_prominence=min_prom,
                    minimum_internal_width=min_int_width,
                    maximum_internal_width=max_int_width,
                    partial_threshold=partial_threshold_l,
                )
                self.blinking_r, self.comp_partial_threshold_r = blinking.peaks(
                    time_series=self.ear_r,
                    minimum_distance=min_dist,
                    minimum_prominence=min_prom,
                    minimum_internal_width=min_int_width,
                    maximum_internal_width=max_int_width,
                    partial_threshold=partial_threshold_r,
                )
            elif self.algorith_extract_tabs.currentWidget() == self.jespbm:
                min_prom = self.getf("JESPBM_min_prom", 0.05)
                window_size = self.geti("JESPBM_window_size", 60)

                diag_running = jwidgets.JDialogRunning()
                diag_running.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)

                self.blinking_l, self.comp_partial_threshold_l = blinking.peaks_espbm(
                    time_series=self.ear_l,
                    minimum_prominence=min_prom,
                    window_size=window_size,
                    partial_threshold=partial_threshold_l,
                    f_min=diag_running.setMinimum,
                    f_max=diag_running.setMaximum,
                    f_val=diag_running.setValue,
                )
                self.blinking_r, self.comp_partial_threshold_r = blinking.peaks_espbm(
                    time_series=self.ear_r,
                    minimum_prominence=min_prom,
                    window_size=window_size,
                    partial_threshold=partial_threshold_r,
                    f_min=diag_running.setMinimum,
                    f_max=diag_running.setMaximum,
                    f_val=diag_running.setValue,
                )
                diag_running.close()
        except Exception as e:
            logger.error("Error while computing the intervals for eye blinking extraction", error=e)
            QMessageBox.critical(None, "Error Blinking Extraction", f"Blinking Extraction Warning Error: {e}")
            return False

        if self.comp_partial_threshold_l is np.nan or self.comp_partial_threshold_r is np.nan:
            QMessageBox.information(
                None,
                "Blinking Extraction Information",
                "We could not compute a automatic threshold based on the data. Continued with default `complete` label or run with sepecific thresholds. We recommend 0.18 for partial blinks.",
            )
        self.progress.setValue(50)

        if self.blinking_l is None or self.blinking_r is None:
            return False

        # check if the blinking data frames are empty
        if self.blinking_l.empty or self.blinking_r.empty:
            QMessageBox.warning(None, "Blinking Extraction Warning", "No blinks found in the data. Please check the data or settings and try again.")
            return False

        try:
            tolerance = self.get("maximum_matching_dist") or 30  # default values
            self.blinking_matched = blinking.match(blinking_l=self.blinking_l, blinking_r=self.blinking_r, tolerance=tolerance)
        except ValueError as e:
            logger.error("Error while matching the blinking data frames", error=e)
            QMessageBox.critical(self, title="Blinking Extraction Error", text="The blinking could not be matched, likely none found")
            return False
        return True

    def plot_intervals(self) -> bool:
        """
        Plot the intervals of eye blinking.

        Returns:
            bool: True if the plotting is successful, False otherwise.
        """
        if self.blinking_l is None or self.blinking_r is None:
            return False

        try:
            self.curve_l.clear()
            self.curve_r.clear()
            self.curve_l.setData(self.ear_l)
            self.curve_r.setData(self.ear_r)

            # TODO add some kind of settings for the colors
            self.scatter_l_comp.clear()
            self.scatter_r_comp.clear()
            self.scatter_l_part.clear()
            self.scatter_r_part.clear()

            # get where the complete blinks are
            x = self.blinking_l[self.blinking_l["blink_type"] == "complete"]["apex_frame"].to_numpy()
            y = self.blinking_l[self.blinking_l["blink_type"] == "complete"]["ear_score"].to_numpy()
            self.scatter_l_comp.setData(x=x, y=y, symbol="o", pen={"color": "#00F", "width": 2})
            x = self.blinking_r[self.blinking_r["blink_type"] == "complete"]["apex_frame"].to_numpy()
            y = self.blinking_r[self.blinking_r["blink_type"] == "complete"]["ear_score"].to_numpy()
            self.scatter_r_comp.setData(x=x, y=y, symbol="o", pen={"color": "#F00", "width": 2})

            # get where the partial blinks are
            x = self.blinking_l[self.blinking_l["blink_type"] == "partial"]["apex_frame"].to_numpy()
            y = self.blinking_l[self.blinking_l["blink_type"] == "partial"]["ear_score"].to_numpy()
            self.scatter_l_part.setData(x=x, y=y, symbol="t", pen={"color": "#00F", "width": 2})
            x = self.blinking_r[self.blinking_r["blink_type"] == "partial"]["apex_frame"].to_numpy()
            y = self.blinking_r[self.blinking_r["blink_type"] == "partial"]["ear_score"].to_numpy()
            self.scatter_r_part.setData(x=x, y=y, symbol="t", pen={"color": "#F00", "width": 2})
        except Exception as e:
            logger.error("Error while plotting the blinking intervals", error=e)
            QMessageBox.critical(None, "Blinking Plotting Error", f"The blinking intervals could not be plotted. {e}")
            return False

        return True

    def clear_on_new_file(self) -> None:
        """
        Clears the variables and UI elements when a new file is loaded.
        """
        self.raw_ear_l = None
        self.raw_ear_r = None
        self.blinking_l = None
        self.blinking_r = None
        self.data_frame = None
        self.data_frame_columns = []
        self.x_lim_max = 1000

        self.comb_ear_l.clear()
        self.comb_ear_r.clear()

        self.table_matched.reset()

        self.curve_l.clear()
        self.curve_r.clear()
        self.scatter_l_comp.clear()
        self.scatter_r_comp.clear()
        self.scatter_l_part.clear()
        self.scatter_r_part.clear()

        self.table_summary.reset()

        self.disable_column_selection()
        self.disable_algorithm()
        self.disable_export()

    def highlight_blink(self, index: int) -> None:
        """
        Highlights the blink at the specified index.

        Args:
            index (int): The index of the blink to highlight.
        """
        if self.blinking_l is None or self.blinking_r is None:
            return
        if self.blinking_matched is None:
            return

        # TODO we already assume that blinking left and right are synced
        frame_left = self.blinking_matched["left"]["apex_frame_og"].iloc[index]
        frame_right = self.blinking_matched["right"]["apex_frame_og"].iloc[index]
        if np.isnan(frame_left) and np.isnan(frame_right):
            return
        if np.isnan(frame_left):
            frame_idx = frame_right
        elif np.isnan(frame_right):
            frame_idx = frame_left
        else:
            frame_idx = min(frame_left, frame_right)
        # show 1 second before and after the blink
        self.graph.setXRange(frame_idx - self.get_selected_fps(), frame_idx + self.get_selected_fps())

        logger.info("Highlighting blink", index=index, frame_idx=frame_idx)
        self.face_preview.set_frame(frame_idx)

    # summary of the results
    def compute_summary(self) -> None:
        """
        Computes the summary of blinking data and updates the UI with the results.
        """
        if self.blinking_matched is None:
            return
        if self.ear_l is None or self.ear_r is None:
            return
        if self.comp_partial_threshold_l is None or self.comp_partial_threshold_r is None:
            return

        fps = self.get_selected_fps()
        self.blinking_summary = blinking.summarize(
            ear_l=self.ear_l,
            ear_r=self.ear_r,
            matched_blinks=self.blinking_matched,
            fps=fps,
            partial_threshold_l=self.comp_partial_threshold_l,
            partial_threshold_r=self.comp_partial_threshold_r,
        )

        self.table_summary.set_data(self.blinking_summary)
        logger.info("Summary computed")

        image = blinking.visualize(self.blinking_matched, fps=fps)
        self.summary_visual_image.setImage(image)
        logger.info("Visual summary computed")

        self.tab_widget_results.setCurrentIndex(1)

    # saving of the results
    def save_results(self) -> None:
        """
        Save the blinking results to a file.

        This method saves the blinking results to a file in either CSV or Excel format.
        The blinking results are obtained from the data frame and the annotations from the table.
        The saved file includes the matched blinking results, summary (if available), and separate sheets for left and right blinking.

        Returns:
            None
        """
        if self.data_frame is None or self.file is None:
            jwidgets.JDialogWarn("Blinking Extraction Error", "No file loaded", "Please load a file and process it first")
            return

        # add the annotations to the data frame
        # TODO this should later be done in the backend and not here!!!
        try:
            overwrite = self.get("overwrite_export") or False
            blinking.save_results(
                self.file,
                self.blinking_l,
                self.blinking_r,
                self.blinking_matched,
                self.blinking_summary,
                format=self.format_export.currentText().lower(),
                exists_ok=overwrite,
            )

        except ValueError as e:
            logger.error("Error while saving the blinking results", error=e)
            jwidgets.JDialogWarn("Blinking Extraction Error", "The blinking results could not be saved", "Please try again")
            return
        except PermissionError as e:
            logger.error("Error while saving the blinking results", error=e)
            jwidgets.JDialogWarn("Blinking Extraction Error", "The blinking results could not be saved", "You have not permission to save the file")
            return
        except FileExistsError as e:
            logger.error("Error while saving the blinking results", error=e)
            jwidgets.JDialogWarn("Blinking Extraction Error", "The blinking results could not be saved", "The file already exists")
            return

        # TODO give user feedback that saving was successful
        logger.info("Saving blinking finished")

    ## general widget functions
    def shut_down(self) -> None:
        """
        Shuts down the widget.

        This method is responsible for shutting down the widget and performing any necessary cleanup operations.

        Parameters:
            None

        Returns:
            None
        """
        # this widget doesn't have any shut down requirements
        self.save()

    def dragEnterEvent(self, event: QtGui.QDropEvent):  # type: ignore # noqa
        """
        Handles the drag enter event for the widget.

        Parameters:
        - event (QtGui.QDropEvent): The drag enter event.

        Returns:
        None
        """
        logger.info("User started dragging event", widget=self)
        mimedata = event.mimeData()
        if mimedata is None:
            return

        if mimedata.hasUrls():
            event.accept()
            logger.info("User started dragging event with mime file", widget=self)
        else:
            event.ignore()
            logger.info("User started dragging event with invalid mime file", widget=self)

    def dropEvent(self, event: QtGui.QDropEvent):  # type: ignore # noqa
        """
        Handle the drop event when files are dropped onto the widget.

        Args:
            event (QtGui.QDropEvent): The drop event object.

        Returns:
            None
        """
        mimedata = event.mimeData()
        if mimedata is None:
            return

        files = [u.toLocalFile() for u in mimedata.urls()]

        if len(files) == 0:
            return

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
