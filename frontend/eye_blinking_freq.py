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

def to_MM_SS(value):
    return f"{int(value / 60):02d}:{int(value % 60):02d}"

def sec_to_min(seconds: float) -> str:
    minutes = int(seconds / 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


# TODO just make this a normal widget and not a splitter
class EyeBlinkingFreq(QtWidgets.QSplitter, config.Config):
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

        self.blinking_table = jwidgets.JBlinkingTable()
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

        le_th_l = QtWidgets.QLineEdit()
        le_th_r = QtWidgets.QLineEdit()
        # le_fps = QtWidgets.QLineEdit()
        le_distance = QtWidgets.QLineEdit()
        le_prominence = QtWidgets.QLineEdit()
        le_width_min = QtWidgets.QLineEdit()
        le_width_max = QtWidgets.QLineEdit()

        MAPPER_FLOAT_STR = (lambda x: float(x), lambda x: str(x))
        MAPPER_INT_STR = (lambda x: int(x), lambda x: str(x))

        self.add_handler("threshold_l", le_th_l, mapper=MAPPER_FLOAT_STR, default=0.16)
        self.add_handler("threshold_r", le_th_r, mapper=MAPPER_FLOAT_STR, default=0.16)
        # self.add_handler("fps", le_fps, mapper=MAPPER_INT_STR, default=240)
        self.add_handler("min_dist", le_distance, mapper=MAPPER_INT_STR, default=50)
        self.add_handler("min_prominence", le_prominence, mapper=MAPPER_FLOAT_STR, default=0.1)
        self.add_handler("min_width", le_width_min, mapper=MAPPER_INT_STR, default=10)
        self.add_handler("max_width", le_width_max, mapper=MAPPER_INT_STR, default=100)

        # le_fps.setValidator(QtGui.QIntValidator())
        le_width_min.setValidator(QtGui.QIntValidator())
        le_width_max.setValidator(QtGui.QIntValidator())

        self.set_algo.addRow("Threshold Left", le_th_l)
        self.set_algo.addRow("Threshold Right", le_th_r)
        # self.set_algo.addRow("FPS", le_fps)
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
        self.add_handler("smooth_size", le_smooth_size, mapper=MAPPER_INT_STR, default=91)
        self.add_handler("smooth_poly", le_smooth_poly, mapper=MAPPER_INT_STR, default=5)
        le_smooth_size.setValidator(QtGui.QIntValidator())
        le_smooth_poly.setValidator(QtGui.QIntValidator())

        box_smooth_layout.addRow("Polynomial Degree", le_smooth_poly)
        box_smooth_layout.addRow("Window Size", le_smooth_size)

        self.set_algo.addRow(box_smooth)

        # Visual Settings #
        self.box_visuals = QtWidgets.QGroupBox("Visual Settings")
        self.set_visuals = QtWidgets.QFormLayout()

        cb_as_time = QtWidgets.QCheckBox()
        cb_width_height = QtWidgets.QCheckBox()
        cb_simple_draw = QtWidgets.QCheckBox()
        btn_reset_graph = QtWidgets.QPushButton(qta.icon("msc.refresh"), "Reset Graph Y Range")
        btn_reset_view = QtWidgets.QPushButton(qta.icon("msc.refresh"), "View Full Graph")

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
        btn_reset_view.clicked.connect(lambda: self.graph.autoRange())

        self.set_visuals.addRow("FPS", self.fps_box)
        # self.set_visuals.addRow("X-Axis As Time", cb_as_time)
        # self.set_visuals.addRow("Draw Width/Height", cb_width_height)
        # self.set_visuals.addRow("Simple Draw", cb_simple_draw)
        self.set_visuals.addRow(btn_reset_view)
        self.set_visuals.addRow(btn_reset_graph)

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

    # loading of the file
    def load_dialog(self) -> None:
        logger.info("Open file dialo for loading CSV file")
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

        self.graph.setLimits(xMin=0, yMin=0, xMax=self.x_lim_max, yMax=1)
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
        assert self.data_frame is not None, "Somehow the data frame is None"
        assert self.data_frame_columns is not None, "Somehow the data frame columns are None"

        assert self.raw_ear_r is not None, "Somehow the raw ear right is None"
        assert self.raw_ear_l is not None, "Somehow the raw ear left is None"
        
        # check if the column selection index are not the same
        if self.comb_ear_l.currentIndex() == self.comb_ear_r.currentIndex():
            dialog = QtWidgets.QMessageBox()
            dialog.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            dialog.setWindowTitle("Blinking Extraction Error")
            dialog.setText("It looks like you selected the same column for both eyes")
            dialog.setInformativeText("Please change your settings and try again")
            dialog.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
            dialog.exec()
            return False

        kwargs = {}
        # kwargs["fps"] = self.get("fps")
        kwargs["minimum_distance"] = self.get("min_dist")
        kwargs["minimum_prominence"] = self.get("min_prominence")
        kwargs["minimum_internal_width"] = self.get("min_width")
        kwargs["maximum_internal_width"] = self.get("max_width")

        do_smoothing: bool = self.get("smooth") or False
        smooth_size: int = self.get("smooth_size") or 91
        smooth_poly: int = self.get("smooth_poly") or 5
        
        self.ear_l = blinking.smooth(self.raw_ear_l, smooth_size, smooth_poly) if do_smoothing else self.raw_ear_l
        self.ear_r = blinking.smooth(self.raw_ear_r, smooth_size, smooth_poly) if do_smoothing else self.raw_ear_r

        self.progress.setValue(40)

        threshold_l = self.get("threshold_l")
        threshold_r = self.get("threshold_r")

        self.blinking_l = blinking.peaks(self.ear_l, threshold=threshold_l, **kwargs)
        self.blinking_r = blinking.peaks(self.ear_r, threshold=threshold_r, **kwargs)
        try:
            self.blinking_matched = blinking.match(self.blinking_l, self.blinking_r, tolerance=30)
        except ValueError as e:
            logger.error("Error while matching the blinking data frames", error=e)
            # create a warning dialog
            dialog = QtWidgets.QMessageBox()
            dialog.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            dialog.setWindowTitle("Blinking Extraction Error")
            dialog.setText("Your extraction settings did not yield any blinkings")
            dialog.setInformativeText("Please change your settings and try again")
            dialog.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
            dialog.exec()
            return False
        return True

    def plot_intervals(self) -> bool:
        if self.blinking_l is None or self.blinking_r is None:
            return False
        
        self.plot_curve_ear_l.clear()
        self.plot_curve_ear_r.clear()
        self.plot_curve_ear_l.setData(self.ear_l)
        self.plot_curve_ear_r.setData(self.ear_r)

        # TODO add some kind of settings for the colors
        self.plot_scatter_blinks_l.clear()
        self.plot_scatter_blinks_r.clear()

        self.plot_scatter_blinks_l.setData(x=self.blinking_l["frame"].to_numpy(), y=self.blinking_l["score"].to_numpy(), pen={"color": "#00F", "width": 2})
        self.plot_scatter_blinks_r.setData(x=self.blinking_r["frame"].to_numpy(), y=self.blinking_r["score"].to_numpy(), pen={"color": "#F00", "width": 2})

        return True

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
        if self.blinking_l is None or self.blinking_r is None:
            return
        if self.blinking_matched is None:
            return

        # TODO we already assume that blinking left and right are synced
        frame_left  = self.blinking_matched["left"]["frame_og"].iloc[index]
        frame_right = self.blinking_matched["right"]["frame_og"].iloc[index]
        if np.isnan(frame_left) and np.isnan(frame_right):
            return
        if np.isnan(frame_left):
            frame_idx = frame_right
        elif np.isnan(frame_right):
            frame_idx = frame_left
        else:
            frame_idx = min(frame_left, frame_right)
        # get fps
        fps = 30 if self.radio_30.isChecked() else 240 # TODO make more general in the future
        self.graph.setXRange(frame_idx - fps, frame_idx + fps)
        self.face_preview.set_frame(frame_idx)

    # summary of the results
    def compute_summary(self) -> None:
        fps = 30 if self.radio_30.isChecked() else 240 # TODO make more general in the future        
        
        self.summary_df = blinking.summarize(self.blinking_matched, fps=fps)
        self.te_results_g.setText(tabulate.tabulate(self.summary_df, headers="keys", tablefmt="github"))
        logger.info("Summary computed")
        
        image = blinking.visualize(self.blinking_matched, fps=fps)
        self.summary_visual_image.setImage(image)
        logger.info("Visual summary computed")
        
        self.tab_widget_results.setCurrentIndex(2)

    # saving of the results
    def save_results(self) -> None:
        if self.data_frame is None or self.file is None:
            return
        
        if self.blinking_l is None or self.blinking_r is None:
            logger.error("No blinking results to save", widget=self)
            return
        
        if self.blinking_matched is None:
            logger.error("No matched blinking results to save", widget=self)
            return

        # get the annotations from the table
        annotations = self.blinking_table.get_annotations()
        # add the annotations to the data frame
        self.blinking_matched["annotation"] = annotations

        if self.format_export.currentText() == "CSV":
            self.blinking_matched.to_csv(self.file.parent / (self.file.stem + "_blinking.csv"), index=False, na_rep="NaN")
            
            if self.summary_df is not None:
                self.summary_dfl.to_csv(self.file.parent / (self.file.stem + "_summary.csv"), index=False, na_rep="NaN")
            
        elif self.format_export.currentText() == "Excel":
            exel_file = self.file.parent / (self.file.stem + "_blinking.xlsx")
            logger.info("Saving blinking results", file=exel_file)
            with pd.ExcelWriter(exel_file) as writer:
                # convert the single columns to integers
                self.blinking_matched[("left",  "single")] = self.blinking_matched[("left",  "single")].astype(str)
                self.blinking_matched[("right", "single")] = self.blinking_matched[("right", "single")].astype(str)
                self.blinking_matched.to_excel(writer, sheet_name="Matched", na_rep="NaN")
                
                if self.summary_df is not None:
                    self.summary_df.to_excel(writer, sheet_name="Summary", na_rep="NaN")
                    
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