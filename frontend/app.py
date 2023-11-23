__all__ = ["JeFaPaTo"]

import argparse
import time

import structlog

from PyQt6 import QtGui
from PyQt6.QtWidgets import QMainWindow, QTabWidget, QProgressBar

from frontend import config
from .landmark_extraction import FacialFeatureExtraction
from .eye_blinking_freq import EyeBlinkingExtraction

logger = structlog.get_logger()

class JeFaPaTo(QMainWindow, config.Config):
    def __init__(self, args: argparse.Namespace) -> None:
        """
        Initializes the main application window.

        Args:
            args (argparse.Namespace): Command-line arguments.

        Returns:
            None
        """
        config.Config.__init__(self, "jefapato")
        QMainWindow.__init__(self)
        
        self.setWindowTitle("JeFaPaTo - Jena Facial Palsy Tool")
        self.showMaximized()
        self.setMinimumSize(800, 600)

        self.central_widget = QTabWidget()
        self.setCentralWidget(self.central_widget)

        self.progress_bar = QProgressBar()

        start = time.time()
        self.tab_facial_feature_extraction = FacialFeatureExtraction(self)
        self.central_widget.addTab(self.tab_facial_feature_extraction, "Facial Feature Extraction")
        logger.info("Start-up time", time=time.time() - start, widget=self.tab_facial_feature_extraction.__class__.__name__)

        start = time.time()
        self.tab_eye_blinking_extraction = EyeBlinkingExtraction(self)
        self.central_widget.addTab(self.tab_eye_blinking_extraction, "Eye Blinking Extraction")
        logger.info("Start-up time", time=time.time() - start, widget=self.tab_eye_blinking_extraction.__class__.__name__)

        tab_idx = args.start_tab
        if tab_idx > self.central_widget.count():
            tab_idx = 0
        self.central_widget.setCurrentIndex(tab_idx)

        self.statusBar().addPermanentWidget(self.progress_bar)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        logger.info("Close Event Detected", widget=self)
        logger.info("Shut Down Processes in each Tab")

        self.tab_facial_feature_extraction.shut_down()
        self.tab_eye_blinking_extraction.shut_down()
        logger.info("Shut Down Processes in each Tab complete", widget=self)
        logger.info("Save Config")
        self.save()
        logger.info("Internal Shut Down complete", widget=self)
        super().closeEvent(event)
