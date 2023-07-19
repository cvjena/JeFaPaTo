__all__ = ["JeFaPaTo"]

import argparse
import time

import git
import structlog

from PyQt6 import QtGui
from PyQt6.QtWidgets import QMainWindow, QTabWidget, QProgressBar

from jefapato import config

from .landmark_extraction import LandmarkExtraction
from .widget_eye_blinking_freq import WidgetEyeBlinkingFreq

logger = structlog.get_logger()

class JeFaPaTo(QMainWindow, config.Config):
    def __init__(self, args: argparse.Namespace) -> None:
        config.Config.__init__(self, "jefapato")
        QMainWindow.__init__(self)
        
        self.setWindowTitle("JeFaPaTo - Jena Facial Palsy Tool")
        self.showMaximized()
        self.setMinimumSize(800, 600)

        self.central_widget = QTabWidget()
        self.setCentralWidget(self.central_widget)

        self.progress_bar = QProgressBar()

        self.VERSION = str(git.Repo(search_parent_directories=True).head.object.hexsha[:7]) # type: ignore

        start = time.time()
        self.tab_eye_blinking = LandmarkExtraction(self)
        logger.info("Start Time LandmarkExtraction", time=time.time() - start)

        start = time.time()
        self.tab_eye_blinking_freq = WidgetEyeBlinkingFreq(self)
        logger.info("Start Time WidgetEyeBlinkingFreq", time=time.time() - start)

        self.central_widget.addTab(self.tab_eye_blinking, "Landmark Extraction")
        self.central_widget.addTab(self.tab_eye_blinking_freq, "Eye Blinking Frequency")

        tab_idx = args.start_tab
        if tab_idx > self.central_widget.count():
            tab_idx = 0
        self.central_widget.setCurrentIndex(tab_idx)

        self.statusBar().addPermanentWidget(self.progress_bar)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        logger.info("Close Event Detected", widget=self)
        logger.info("Shut Down Processes in each Tab")

        self.tab_eye_blinking.shut_down()
        self.tab_eye_blinking_freq.shut_down()
        logger.info("Shut Down Processes in each Tab complete", widget=self)
        logger.info("Save Config")
        self.save()
        logger.info("Internal Shut Down complete", widget=self)
        super().closeEvent(event)
