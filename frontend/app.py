__all__ = ["JeFaPaTo"]

import argparse
import time

import structlog

from PyQt6 import QtGui
from PyQt6.QtWidgets import QMainWindow, QTabWidget, QProgressBar, QWidget

from frontend import config
from .ui_facial_feature_extraction import FacialFeatureExtraction
from .ui_eye_blinking_extraction import EyeBlinkingExtraction

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
        
        self.uis: list[QWidget] = [
            FacialFeatureExtraction, 
            EyeBlinkingExtraction,
        ]
        self.tabs: list[QWidget] = []
        
        for ui in self.uis:
            start = time.time()        
            temp = ui(self)
            self.tabs.append(temp)
            self.central_widget.addTab(temp, temp.__class__.__name__)
            logger.info("Start-up time", time=time.time() - start, widget=temp.__class__.__name__)
    
        tab_idx = args.start_tab
        if tab_idx > self.central_widget.count():
            tab_idx = 0
        self.central_widget.setCurrentIndex(tab_idx)

        self.statusBar().addPermanentWidget(self.progress_bar)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """
        Event handler for the close event of the application window.
        Performs necessary cleanup tasks before closing the application.
        
        Args:
            event (QtGui.QCloseEvent): The close event object.
        """
        
        logger.info("Close Event Detected", widget=self)
        logger.info("Shut Down Processes in each Tab")

        for tab in self.tabs:
            tab.shut_down()

        logger.info("Shut Down Processes in each Tab complete", widget=self)
        logger.info("Save Config")
        self.save()
        logger.info("Internal Shut Down complete", widget=self)
        super().closeEvent(event)
