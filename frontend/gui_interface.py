__all__ = ["JeFaPaToGUISignalThread"]

from collections import OrderedDict
from typing import Any

import numpy as np
from PyQt6 import QtCore, QtWidgets

from jefapato import facial_features


class JeFaPaToGUISignalThread(QtCore.QThread):
    sig_updated_display = QtCore.pyqtSignal(np.ndarray)
    sig_updated_feature = QtCore.pyqtSignal(dict)
    sig_processed_percentage = QtCore.pyqtSignal(float)
    sig_started = QtCore.pyqtSignal()
    sig_paused = QtCore.pyqtSignal()
    sig_resumed = QtCore.pyqtSignal()
    sig_finished = QtCore.pyqtSignal()
    
    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent=parent)
        self.parent = parent
        self.stopped = False

        self.update_counter = 0
        self.update_interval = 10

    def run(self):
        while True:
            if self.stopped:
                return
            self.msleep(100)

    def stop(self):
        self.stopped = True
        
    def set_update_interval(self, interval: int):
        self.update_interval = interval

    @facial_features.FaceAnalyzer.hookimpl
    def updated_display(self, image: np.ndarray):
        self.update_counter += 1
        
        if self.update_counter % self.update_interval == 0:
            self.update_counter = 0
            self.sig_updated_display.emit(image)

    @facial_features.FaceAnalyzer.hookimpl
    def updated_feature(self, feature_data: OrderedDict[str, Any]) -> None:
        self.sig_updated_feature.emit(feature_data)
        
    @facial_features.FaceAnalyzer.hookimpl
    def processed_percentage(self, percentage: float) -> None:
        self.sig_processed_percentage.emit(percentage)
    
    @facial_features.FaceAnalyzer.hookimpl
    def started(self) -> None:
        self.sig_started.emit()
    
    @facial_features.FaceAnalyzer.hookimpl
    def paused(self) -> None:
        self.sig_paused.emit()
    
    @facial_features.FaceAnalyzer.hookimpl
    def resumed(self) -> None:
        self.sig_resumed.emit()

    @facial_features.FaceAnalyzer.hookimpl
    def finished(self) -> None:
        self.sig_finished.emit()