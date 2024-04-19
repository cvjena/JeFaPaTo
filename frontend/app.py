__all__ = ["JeFaPaTo"]

import argparse
import re
import time

import structlog

from PyQt6 import QtGui
from PyQt6.QtWidgets import QMainWindow, QTabWidget, QProgressBar, QWidget, QMessageBox

from frontend import config
from .ui_facial_feature_extraction import FacialFeatureExtraction
from .ui_eye_blinking_extraction import EyeBlinkingExtraction

logger = structlog.get_logger()

def add_space_between_words(text):
    return re.sub(r"(\w)([A-Z])", r"\1 \2", text)

LICENCE_TEXT = """
JeFaPaTo is licensed under the MIT License.
     
MIT License

Copyright (c) [2023] [Tim Büchner]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

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
            self.central_widget.addTab(temp, add_space_between_words(temp.__class__.__name__))
            logger.info("Start-up time", time=time.time() - start, widget=temp.__class__.__name__)
    
        tab_idx = args.start_tab
        if tab_idx > self.central_widget.count():
            tab_idx = 0
        self.central_widget.setCurrentIndex(tab_idx)

        self.statusBar().addPermanentWidget(self.progress_bar)

        # Create menu bar
        menu_bar = self.menuBar()
        # Create Help menu
        help_menu = menu_bar.addMenu("Help")

        help_menu.addAction("About", self.show_about)
        help_menu.addAction("Documentation", self.show_documentation)
        help_menu.addAction("License", self.show_license)
        help_menu.addAction("Acknowledgements", self.show_acknowledgements)

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

    def show_about(self):
        """
        Shows the About dialog.
        """
        ## use a dialog to show the about information
        dialog = QMessageBox()
        dialog.setWindowTitle("About JeFaPaTo")
        dialog.setText("JeFaPaTo - Jena Facial Palsy Tool")
        dialog.setInformativeText("""
            <ul>
            <li> Version: 1.0.0 </li>
            <li> Author: Tim Büchner </li>
            <li> License: MIT </li>
            <li> Link: <a href="https://github.com/cvjena/JeFaPaTo">Website</a> </li>
            </ul>
        """)
        dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
        dialog.exec()
        
    def show_documentation(self):
        """
        Shows the Documentation dialog.
        """
        dialog = QMessageBox()
        dialog.setWindowTitle("Documentation")
        dialog.setText("Documentation")
        dialog.setInformativeText("""
            Documentation is available at <a href="https://github.com/cvjena/JeFaPaTo/wiki"> our wiki page</a>.
        """
        )
        dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
        dialog.exec()
        
    def show_license(self):
        """
        Shows the License dialog.
        """
        dialog = QMessageBox()
        dialog.setWindowTitle("License")
        dialog.setText("License")
        dialog.setInformativeText(LICENCE_TEXT)
        dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
        dialog.exec()
        
    def show_acknowledgements(self):
        """
        Shows the Acknowledgements dialog.
        """
    
        dialog = QMessageBox()
        dialog.setWindowTitle("Acknowledgements")
        dialog.setText("Acknowledgements")
        dialog.setInformativeText("""
            JeFaPaTo is based on the <a href="https://github.com/google/mediapipe"> MediaPipe library </a> by Google.
            We would like to thank the developers for their great work and the possibility to use their library. 
            Additionally, we would like to thank the <a href="https://opencv.org/"> OpenCV </a> team for their great work and the possibility to use their library.
            Also, we thank our medical partners for their support and feedback.
        """)
        dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
        dialog.exec()
