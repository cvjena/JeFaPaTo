import logging
from pathlib import Path

from PyQt5.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from jefapato.analyser import EyeBlinkingVideoAnalyser
from jefapato.plotter import EyeDetailWidget, FrameWidget, GraphWidget


class WidgetEyeBlinking(QSplitter):
    def __init__(self):
        super().__init__()
        self.video_file_path: Path = None

        self.logger = logging.getLogger("eyeBlinkingDetection")

        self.widget_frame = FrameWidget()
        self.widget_detail = EyeDetailWidget()
        self.widget_graph = GraphWidget()

        widget_l = QWidget()
        widget_r = QWidget()

        self.vlayout_ls = QVBoxLayout()
        self.vlayout_rs = QVBoxLayout()
        self.flayout_se = QFormLayout()

        widget_l.setLayout(self.vlayout_ls)
        widget_r.setLayout(self.vlayout_rs)

        self.addWidget(widget_l)
        self.addWidget(widget_r)

        self.le_threshold = QLineEdit()
        self.le_bmp_l = QLineEdit()
        self.le_bmp_r = QLineEdit()

        self.cb_anal = QCheckBox()
        self.bt_open = QPushButton("Open Video File")
        self.bt_anal = QPushButton("Analyze Video")
        self.bt_anal_stop: QPushButton = QPushButton("Cancel")

        self.pb_anal = QProgressBar()
        self.face_skipper = QSpinBox()
        self.frame_skipper = QSpinBox()

        self.vlayout_ls.addWidget(self.widget_frame)
        self.vlayout_ls.addWidget(self.widget_graph)

        self.vlayout_rs.addWidget(self.widget_detail)
        self.vlayout_rs.addLayout(self.flayout_se)

        self.flayout_se.addRow(self.bt_open)
        self.flayout_se.addRow(self.bt_anal)
        self.flayout_se.addRow(self.bt_anal_stop)
        self.flayout_se.addRow("Skip Frame for Face Detection:", self.face_skipper)
        self.flayout_se.addRow("Skip Frames For Display:", self.frame_skipper)
        self.flayout_se.addRow("Progress:", self.pb_anal)
        self.flayout_se.addRow("Threshold:", self.le_threshold)
        self.flayout_se.addRow("Blinking Rate Left:", self.le_bmp_l)
        self.flayout_se.addRow("Blinking Rate Right:", self.le_bmp_r)

        self.ea = EyeBlinkingVideoAnalyser(
            self.widget_frame,
            self.widget_detail,
            self.widget_graph,
            self.le_threshold,
        )

        self.ea.connect_on_started([self.gui_analysis_start, self.pb_anal.reset])
        self.ea.connect_on_finished(
            [self.gui_analysis_finished, self.compute_blinking_per_minute]
        )
        self.ea.connect_processed_percentage([self.pb_anal.setValue])

        self.bt_open.clicked.connect(self.load_video)
        self.bt_anal.clicked.connect(self.ea.analysis_start)
        self.bt_anal_stop.clicked.connect(self.ea.stop)

        self.face_skipper.setRange(3, 20)
        self.face_skipper.setValue(5)
        self.face_skipper.valueChanged.connect(self.ea.set_face_detect_skip)

        self.frame_skipper.setRange(1, 20)
        self.frame_skipper.setValue(5)
        self.frame_skipper.valueChanged.connect(self.ea.set_frame_skip)

        # disable analyse button and check box
        self.bt_anal.setDisabled(True)
        self.cb_anal.setDisabled(True)
        self.bt_anal_stop.setDisabled(True)

        self.le_bmp_r.setReadOnly(True)
        self.le_bmp_l.setReadOnly(True)
        self.le_threshold.setReadOnly(True)

        self.ea.set_threshold(0.20)

        self.setStretchFactor(0, 7)
        self.setStretchFactor(1, 3)

    def compute_blinking_per_minute(self):
        self.bpm_l, self.bpm_r = self.ea.blinking_rate()
        self.le_bmp_l.setText(f"{self.bpm_l:5.2f}")
        self.le_bmp_r.setText(f"{self.bpm_r:5.2f}")

    def gui_analysis_start(self):
        self.bt_open.setDisabled(True)
        self.bt_anal.setDisabled(True)
        self.bt_anal_stop.setDisabled(False)
        self.cb_anal.setDisabled(True)
        self.le_threshold.setDisabled(True)
        self.face_skipper.setDisabled(True)
        self.frame_skipper.setDisabled(True)

    def gui_analysis_finished(self):
        self.bt_open.setDisabled(False)
        self.bt_anal.setText("Analyze Video")
        self.bt_anal.setDisabled(False)
        self.bt_anal_stop.setDisabled(True)

        self.cb_anal.setDisabled(False)
        self.le_threshold.setDisabled(False)
        self.face_skipper.setDisabled(False)
        self.frame_skipper.setDisabled(False)

    def load_video(self):
        self.logger.info("Open file explorer")
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "Select video file",
            ".",
            "Video Files (*.mp4 *.flv *.ts *.mts *.avi *.mov)",
        )

        if fileName != "":
            self.logger.info(f"Load video file: {fileName}")
            self.video_file_path = Path(fileName)

            self.ea.set_resource_path(self.video_file_path)
            self.bt_anal.setDisabled(False)
            self.cb_anal.setDisabled(False)
        else:
            self.logger.info("No video file was selected")
