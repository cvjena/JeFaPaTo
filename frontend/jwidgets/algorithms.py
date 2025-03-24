__all__ = [
    "JPeaks",
    "JSmoothing",
    "JESPBM",
    "JBlinkingAnalysis",
]


from PyQt6 import QtCore, QtGui
from PyQt6.QtWidgets import QComboBox, QFormLayout, QGridLayout, QGroupBox, QLabel, QLineEdit, QMessageBox, QPushButton, QWidget

from frontend import config
from frontend.jwidgets.utils import F2S, I2S, create_help_button
from jefapato import blinking

local = QtCore.QLocale(QtCore.QLocale.Language.English, QtCore.QLocale.Country.UnitedStates)
D_VALID = QtGui.QDoubleValidator()
D_VALID.setBottom(0)
D_VALID.setDecimals(3)
D_VALID.setNotation(QtGui.QDoubleValidator.Notation.StandardNotation)
D_VALID.setLocale(local)

I_VALID = QtGui.QIntValidator()
I_VALID.setBottom(0)
I_VALID.setLocale(local)


class JBlinkingAnalysis(QWidget):
    def __init__(self, config_ref: config.Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid = QGridLayout()

        le_maximum_matching_dist = QLineEdit()
        le_maximum_matching_dist.setValidator(I_VALID)
        config_ref.add_handler("maximum_matching_dist", le_maximum_matching_dist, mapper=I2S, default=30)

        self.grid.addWidget(QLabel("Maximum Matching Distance"), 5, 0)
        self.grid.addWidget(le_maximum_matching_dist, 5, 1)
        self.grid.addWidget(create_help_button("Maximum Matching Distance: The maximum distance between two peaks to be matched in frames. Rec: 15@30FPS, 30@240FPS"), 5, 2)

        le_partial_threshold_left = QLineEdit()
        config_ref.add_handler("partial_threshold_l", le_partial_threshold_left, default="auto")

        self.grid.addWidget(QLabel("Partial Threshold Left"), 6, 0)
        self.grid.addWidget(le_partial_threshold_left, 6, 1)
        self.grid.addWidget(create_help_button("Partial Threshold Left: The threshold for a partial blink in EAR score either 'auto' or a float number. Rec: 0.18"), 6, 2)

        le_partial_threshold_right = QLineEdit()
        config_ref.add_handler("partial_threshold_r", le_partial_threshold_right, default="auto")

        self.grid.addWidget(QLabel("Partial Threshold Right"), 7, 0)
        self.grid.addWidget(le_partial_threshold_right, 7, 1)
        self.grid.addWidget(create_help_button("Partial Threshold Right: The threshold for a partial blink in EAR score either 'auto' or a float number. Rec: 0.18"), 7, 2)

        # TODO this value is not saved in the config!
        self.cb_video_fps = QComboBox()
        self.cb_video_fps.addItems(["24", "30", "60", "120", "240"])
        self.cb_video_fps.setCurrentIndex(4)

        self.grid.addWidget(QLabel("Video FPS"), 8, 0)
        self.grid.addWidget(self.cb_video_fps, 8, 1)
        self.grid.addWidget(create_help_button("Video FPS: The frames per second of the video."), 8, 2)

        self.setLayout(self.grid)
        self.show()


class JSmoothing(QWidget):
    def __init__(self, config_ref: config.Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid = QGridLayout()

        box_smooth = QGroupBox("Smoothing")
        box_smooth.setCheckable(True)
        config_ref.add_handler("smooth", box_smooth)
        box_smooth_layout = QFormLayout()
        box_smooth.setLayout(box_smooth_layout)

        le_smooth_size = QLineEdit()
        le_smooth_size.setValidator(I_VALID)
        config_ref.add_handler("smooth_size", le_smooth_size, mapper=I2S, default=7)
        box_smooth_layout.addRow("Window Size", le_smooth_size)

        le_smooth_poly = QLineEdit()
        le_smooth_poly.setValidator(I_VALID)
        config_ref.add_handler("smooth_poly", le_smooth_poly, mapper=I2S, default=3)
        box_smooth_layout.addRow("Polynomial Degree", le_smooth_poly)

        self.grid.addWidget(box_smooth, 0, 0, 1, 3)
        self.grid.addWidget(create_help_button("Smoothing: The smoothing of the EAR data. Rec: Window Size: 7, Polynomial Degree: 3"), 0, 3)

        self.setLayout(self.grid)
        self.show()


class JPeaks(QWidget):
    def __init__(self, config_ref: config.Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid = QGridLayout()
        self.grid.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        extraction_help_button = QPushButton("Peak Algorithm Help")
        extraction_help_button.clicked.connect(lambda: QMessageBox.information(None, "Help", blinking.HELP_PEAKS))

        self.grid.addWidget(extraction_help_button, 0, 0, 1, 3)

        le_minimum_distance = QLineEdit()
        le_minimum_distance.setValidator(I_VALID)
        config_ref.add_handler("min_dist", le_minimum_distance, mapper=I2S, default=50)

        self.grid.addWidget(QLabel("Minimum Distance"), 1, 0)
        self.grid.addWidget(le_minimum_distance, 1, 1)
        self.grid.addWidget(create_help_button("Minimum Distance: The minimum distance between two peaks in frames. Rec: 10@30FPS, 50@240FPS"), 1, 2)

        le_minimum_prominence = QLineEdit()
        le_minimum_prominence.setValidator(D_VALID)
        config_ref.add_handler("min_prominence", le_minimum_prominence, mapper=F2S, default=0.1)

        self.grid.addWidget(QLabel("Minimum Prominence"), 2, 0)
        self.grid.addWidget(le_minimum_prominence, 2, 1)
        self.grid.addWidget(create_help_button("Minimum Prominence: The minimum prominence of a peak in EAR score. Rec: 0.1"), 2, 2)

        le_minimum_internal_width = QLineEdit()
        le_minimum_internal_width.setValidator(I_VALID)
        config_ref.add_handler("min_width", le_minimum_internal_width, mapper=I2S, default=10)

        self.grid.addWidget(QLabel("Minimum Internal Width"), 3, 0)
        self.grid.addWidget(le_minimum_internal_width, 3, 1)
        self.grid.addWidget(create_help_button("Minimum Internal Width: The minimum width of a peak in frames. Rec: 4@30FPS, 20@240FPS"), 3, 2)

        le_maximum_internal_width = QLineEdit()
        le_maximum_internal_width.setValidator(I_VALID)
        config_ref.add_handler("max_width", le_maximum_internal_width, mapper=I2S, default=100)

        self.grid.addWidget(QLabel("Maximum Internal Width"), 4, 0)
        self.grid.addWidget(le_maximum_internal_width, 4, 1)
        self.grid.addWidget(create_help_button("Maximum Internal Width: The maximum width of a peak in frames. Rec: 20@30FPS, 100@240FPS"), 4, 2)

        # final init
        self.setLayout(self.grid)
        self.show()


class JESPBM(QWidget):
    def __init__(self, config_ref: config.Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid = QGridLayout()
        self.grid.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        extraction_help_button = QPushButton("ESPBM Algorithm Help")
        extraction_help_button.clicked.connect(lambda: QMessageBox.information(None, "Help", blinking.HELP_ESPBM))
        self.grid.addWidget(extraction_help_button, 0, 0, 1, 3)

        le_window_size = QLineEdit()
        le_window_size.setValidator(I_VALID)
        config_ref.add_handler("JESPBM_window_size", le_window_size, mapper=I2S, default=80)

        self.grid.addWidget(QLabel("Window Size"), 1, 0)
        self.grid.addWidget(le_window_size, 1, 1)
        self.grid.addWidget(create_help_button("Window Size: The window size for the ESPBM algorithm. Rec: 80"), 1, 2)

        le_minimum_prominence = QLineEdit()
        le_minimum_prominence.setValidator(D_VALID)
        config_ref.add_handler("JESPBM_min_prom", le_minimum_prominence, mapper=F2S, default=0.1)

        self.grid.addWidget(QLabel("Minimum Prominence"), 2, 0)
        self.grid.addWidget(le_minimum_prominence, 2, 1)
        self.grid.addWidget(create_help_button("Minimum Prominence: The minimum prominence of a peak in EAR score. Rec: 0.1"), 2, 2)

        self.setLayout(self.grid)
        self.show()
