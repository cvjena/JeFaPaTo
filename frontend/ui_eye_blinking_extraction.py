from pathlib import Path

import numpy as np
import pandas as pd
import qtawesome as qta
import structlog
import tabulate
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets

from PyQt6.QtCore import pyqtSignal

from jefapato import blinking
from frontend import config, jwidgets

logger = structlog.get_logger()

DOWNSAMPLE_FACTOR = 8

def to_float(value: str) -> float:
    """
    Converts a string value to a float.

    Args:
        value (str): The string value to be converted.

    Returns:
        float: The converted float value. If the conversion fails, returns 0.0.
    """
    try:
        return float(value)
    except ValueError:
        return 0.0

def to_int(value: str) -> int:
    """
    Converts a string value to an integer.
    
    Args:
        value (str): The string value to be converted.
    
    Returns:
        int: The converted integer value. If the conversion fails, returns 0.
    """
    try:
        return int(value)
    except ValueError:  
        return 0

F2S = (lambda x: to_float(x), lambda x: str(x))
I2S = (lambda x: to_int(x), lambda x: str(x))

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
        self.plot_curve_ear_l = self.graph.add_curve({"color": "#00F", "width": 2}) # TODO add correct colors...
        self.plot_curve_ear_r = self.graph.add_curve({"color": "#F00", "width": 2}) # TODO add correct colors...
        self.plot_scatter_blinks_l = self.graph.add_scatter()
        self.plot_scatter_blinks_r = self.graph.add_scatter()

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

        self.blinking_table = jwidgets.JTableBlinking()
        self.blinking_table.selection_changed.connect(self.highlight_blink)

        # second tab is the text information
        self.te_results_g = QtWidgets.QTextEdit()
        self.te_results_g.setFontFamily("mono")
        self.te_results_g.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.NoWrap)
        
        self.summary_visual_widget = pg.GraphicsLayoutWidget()
        self.summary_visual = pg.ViewBox(invertY=True, lockAspect=True, enableMenu=True, enableMouse=True)
        self.summary_visual_image = pg.ImageItem()
        self.summary_visual.addItem(self.summary_visual_image)
        self.summary_visual_widget.addItem(self.summary_visual)
        self.tab_widget_results.addTab(self.blinking_table, "Blinking Table")
        self.tab_widget_results.addTab(self.te_results_g,   "Summary")
        self.tab_widget_results.addTab(self.summary_visual_widget,  "Visual Summary")

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

        self.progress = self.parent().progress_bar # type: ignore 
        self.btn_load.clicked.connect(self.load_dialog)
        self.btn_anal.clicked.connect(self.extract_blinks)
        self.btn_summ.clicked.connect(self.compute_summary)
        self.btn_eprt.clicked.connect(self.save_results)

        # algorithm settings box
        self.box_settings = QtWidgets.QGroupBox("Algorithm Settings")
        # dont make the groupbox changeable in height
        self.box_settings.setMinimumHeight(200)
        self.set_algo = QtWidgets.QFormLayout()
        
        local = QtCore.QLocale(QtCore.QLocale.Language.English, QtCore.QLocale.Country.UnitedStates) 
        doulbe_validator = QtGui.QDoubleValidator()
        doulbe_validator.setBottom(0)
        doulbe_validator.setDecimals(3)
        doulbe_validator.setNotation(QtGui.QDoubleValidator.Notation.StandardNotation)
        doulbe_validator.setLocale(local)

        int_validator = QtGui.QIntValidator()
        int_validator.setBottom(0)
        int_validator.setLocale(local)
        
        le_threshold_left = QtWidgets.QLineEdit()
        le_threshold_left.setValidator(doulbe_validator)
        self.add_handler("threshold_l", le_threshold_left, mapper=F2S, default=0.16)
        self.set_algo.addRow("Threshold Left", le_threshold_left)
        
        le_threshold_right = QtWidgets.QLineEdit()
        le_threshold_right.setValidator(doulbe_validator)
        self.add_handler("threshold_r", le_threshold_right, mapper=F2S, default=0.16)
        self.set_algo.addRow("Threshold Right", le_threshold_right)

        le_minimum_distance = QtWidgets.QLineEdit()
        le_minimum_distance.setValidator(int_validator)
        self.add_handler("min_dist", le_minimum_distance, mapper=I2S, default=50)
        self.set_algo.addRow("Minimum Distance", le_minimum_distance)

        le_minimum_prominence = QtWidgets.QLineEdit()
        le_minimum_prominence.setValidator(doulbe_validator)
        self.add_handler("min_prominence", le_minimum_prominence, mapper=F2S, default=0.1)
        self.set_algo.addRow("Minimum Prominence", le_minimum_prominence)

        le_minimum_internal_width = QtWidgets.QLineEdit()
        le_minimum_internal_width.setValidator(int_validator)
        self.add_handler("min_width", le_minimum_internal_width, mapper=I2S, default=10)
        self.set_algo.addRow("Mininum Internal Width", le_minimum_internal_width)

        le_maximum_internal_width = QtWidgets.QLineEdit()
        le_maximum_internal_width.setValidator(int_validator)        
        self.add_handler("max_width", le_maximum_internal_width, mapper=I2S, default=100)
        self.set_algo.addRow("Maximum Internal Width", le_maximum_internal_width)

        le_maximum_matching_dist = QtWidgets.QLineEdit()
        le_maximum_matching_dist.setValidator(int_validator)
        self.add_handler("maximum_matching_dist", le_maximum_matching_dist, mapper=I2S, default=30)
        self.set_algo.addRow("Maximum Matching Distance", le_maximum_matching_dist)
        
        # TODO this value is not saved in the config!
        self.cb_video_fps = QtWidgets.QComboBox()
        self.cb_video_fps.addItems(["24", "30", "60", "120", "240"])
        self.cb_video_fps.setCurrentIndex(4)
        self.cb_video_fps.currentIndexChanged.connect(self.compute_graph_axis)
        self.set_algo.addRow("Video FPS", self.cb_video_fps)
        
        box_smooth = QtWidgets.QGroupBox("Smoothing")
        box_smooth.setCheckable(True)
        self.add_handler("smooth", box_smooth)
        box_smooth_layout = QtWidgets.QFormLayout()
        box_smooth.setLayout(box_smooth_layout)

        le_smooth_size = QtWidgets.QLineEdit()
        le_smooth_size.setValidator(int_validator)
        self.add_handler("smooth_size", le_smooth_size, mapper=I2S, default=91)
        box_smooth_layout.addRow("Window Size", le_smooth_size)
        
        le_smooth_poly = QtWidgets.QLineEdit()
        le_smooth_poly.setValidator(int_validator)
        self.add_handler("smooth_poly", le_smooth_poly, mapper=I2S, default=5)
        box_smooth_layout.addRow("Polynomial Degree", le_smooth_poly)

        self.set_algo.addRow(box_smooth)
        
        # Visual Settings #
        self.box_visuals = QtWidgets.QGroupBox("Graph Control")
        self.set_visuals = QtWidgets.QFormLayout()

        self.cb_as_time = QtWidgets.QCheckBox("Show Time")
        self.add_handler("as_time", self.cb_as_time)
        self.set_visuals.addRow(self.cb_as_time)
        self.cb_as_time.stateChanged.connect(self.compute_graph_axis)

        btn_reset_graph = QtWidgets.QPushButton(qta.icon("msc.refresh"), "Reset Graph Y Range")
        btn_reset_graph.clicked.connect(lambda: self.graph.setYRange(0, 1))
        self.set_visuals.addRow(btn_reset_graph)

        btn_reset_view  = QtWidgets.QPushButton(qta.icon("msc.refresh"), "View Full Graph")
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
        return int(self.cb_video_fps.currentText())

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
        self.plot_curve_ear_l.clear()
        self.plot_curve_ear_r.clear()

        if self.raw_ear_l is not None:
            self.plot_curve_ear_l.setData(self.raw_ear_l)
        if self.raw_ear_r is not None:
            self.plot_curve_ear_r.setData(self.raw_ear_r)

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
        if not self.compute_intervals():
            return
        self.progress.setValue(60)

        if not self.plot_intervals():
            return
        self.progress.setValue(80)

        if self.blinking_l is None or self.blinking_r is None:
            logger.error("Somehow the blinking data frames are None")
            return
        if self.blinking_matched is None:
            logger.error("Somehow the matched blinking data frame is None")
            return

        self.blinking_table.set_data(self.blinking_matched)
        self.progress.setValue(100)

        self.enable_export()
        self.tab_widget_results.setCurrentIndex(0)

    def compute_intervals(self) -> None:
        """
        Computes the intervals for eye blinking extraction based on the provided settings.

        Returns:
            None

        Raises:
            AssertionError: If the data frame or data frame columns are None.
            AssertionError: If the raw ear right or raw ear left are None.
            ValueError: If the same column is selected for both eyes.
            ValueError: If the blinking data frames cannot be matched.

        """
        def validate_setting(setting_name: str) -> tuple[bool, int | float]:
            try:
                value = self.get(setting_name)
            except ValueError:
                logger.error("Error while validating the settings", setting=setting_name)
                jwidgets.JDialogWarn("Blinking Extraction Error", f"The setting {setting_name} is not a valid input.", "Please change your settings and try again")
                return False, None
            return True, value

        # check if the column selection index are not the same
        if self.comb_ear_l.currentIndex() == self.comb_ear_r.currentIndex():
            logger.error("The same column is selected for both eyes")
            jwidgets.JDialogWarn("Blinking Extraction Error", "Both EAR columns are the same!", "Please select different columns and try again",)
            return False
        
        succ, threshold_l = validate_setting("threshold_l")
        if not succ:
            return False
        succ, threshold_r = validate_setting("threshold_r")
        if not succ:
            return False
        succ, minimum_distance = validate_setting("min_dist")
        if not succ:
            return False
        succ, minimum_prominence = validate_setting("min_prominence")
        if not succ:
            return False
        succ, minimum_internal_width = validate_setting("min_width")
        if not succ:
            return False
        succ, maximum_matching_dist = validate_setting("maximum_matching_dist")
        if not succ:
            return False
        succ, maximum_internal_width = validate_setting("max_width")
        if not succ:
            return False
        succ, smooth_size = validate_setting("smooth_size")
        if not succ:
            return False
        succ, smooth_poly = validate_setting("smooth_poly")
        if not succ:
            return False

        self.ear_l = blinking.smooth(self.raw_ear_l, smooth_size, smooth_poly) if self.get("smooth") else self.raw_ear_l
        self.ear_r = blinking.smooth(self.raw_ear_r, smooth_size, smooth_poly) if self.get("smooth") else self.raw_ear_r

        self.progress.setValue(40)

        threshold_l = self.get("threshold_l")
        threshold_r = self.get("threshold_r")

        self.blinking_l = blinking.peaks(self.ear_l, threshold_l, minimum_distance, minimum_prominence, minimum_internal_width, maximum_internal_width)
        self.blinking_r = blinking.peaks(self.ear_r, threshold_r, minimum_distance, minimum_prominence, minimum_internal_width, maximum_internal_width)
        
        try:
            self.blinking_matched = blinking.match(self.blinking_l, self.blinking_r, tolerance=maximum_matching_dist)
        except ValueError as e:
            logger.error("Error while matching the blinking data frames", error=e)
            jwidgets.JDialogWarn( "Blinking Extraction Error", "The blinking could not be matched, likely none found", "Please change your settings and try again")
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
        
        self.plot_curve_ear_l.clear()
        self.plot_curve_ear_r.clear()
        self.plot_curve_ear_l.setData(self.ear_l)
        self.plot_curve_ear_r.setData(self.ear_r)

        # TODO add some kind of settings for the colors
        self.plot_scatter_blinks_l.clear()
        self.plot_scatter_blinks_r.clear()

        self.plot_scatter_blinks_l.setData(x=self.blinking_l["apex_frame"].to_numpy(), y=self.blinking_l["ear_score"].to_numpy(), pen={"color": "#00F", "width": 2})
        self.plot_scatter_blinks_r.setData(x=self.blinking_r["apex_frame"].to_numpy(), y=self.blinking_r["ear_score"].to_numpy(), pen={"color": "#F00", "width": 2})

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

        self.blinking_table.reset()

        self.plot_curve_ear_l.clear()
        self.plot_curve_ear_r.clear()
        self.plot_scatter_blinks_l.clear()
        self.plot_scatter_blinks_r.clear()

        self.te_results_g.setText("")

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
        frame_left  = self.blinking_matched["left"]["apex_frame_og"].iloc[index]
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
        self.face_preview.set_frame(frame_idx)

    # summary of the results
    def compute_summary(self) -> None:
        """
        Computes the summary of blinking data and updates the UI with the results.
        """
        fps = self.get_selected_fps()
        self.blinking_summary = blinking.summarize(self.blinking_matched, fps=fps)
        self.te_results_g.setText(tabulate.tabulate(self.blinking_summary, headers="keys", tablefmt="github"))
        logger.info("Summary computed")
        
        image = blinking.visualize(self.blinking_matched, fps=fps)
        self.summary_visual_image.setImage(image)
        logger.info("Visual summary computed")
        
        self.tab_widget_results.setCurrentIndex(2)

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
        if self.blinking_matched is None:
            self.blinking_matched["annotation"] = self.blinking_table.get_annotations()
        
        try:
            blinking.save_results(
                self.file,
                self.blinking_l,
                self.blinking_r,
                self.blinking_matched,
                self.blinking_summary,
                format=self.format_export.currentText().lower(),
                exists_ok=self.get("overwrite_export"),
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

    def dragEnterEvent(self, event: QtGui.QDropEvent):
        """
        Handles the drag enter event for the widget.

        Parameters:
        - event (QtGui.QDropEvent): The drag enter event.

        Returns:
        None
        """
        logger.info("User started dragging event", widget=self)
        if event.mimeData().hasUrls():
            event.accept()
            logger.info("User started dragging event with mime file", widget=self)
        else:
            event.ignore()
            logger.info("User started dragging event with invalid mime file", widget=self)

    def dropEvent(self, event: QtGui.QDropEvent):
        """
        Handle the drop event when files are dropped onto the widget.

        Args:
            event (QtGui.QDropEvent): The drop event object.

        Returns:
            None
        """
        files = [u.toLocalFile() for u in event.mimeData().urls()]

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