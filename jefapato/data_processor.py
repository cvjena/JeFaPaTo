from typing import Callable, Any
from .data_loader import DataLoader

import numpy as np

from PyQt5.QtCore import QThread, pyqtSignal

class DataProcessor(QThread):
    processingStarted:  pyqtSignal = pyqtSignal()
    processingUpdated:  pyqtSignal = pyqtSignal(np.ndarray, int)
    processingFinished: pyqtSignal = pyqtSignal()

    def __init__(self, analyse_func: Callable, data_amount: int, data_loader: DataLoader) -> None:
        super().__init__()
        self.analyse_func: Callable = analyse_func
        self.data_loader: DataLoader = data_loader
        self.data_amount: int = data_amount

    def __del__(self):
        self.wait()

    def run(self):
        # TODO add loggin
        self.processingStarted.emit()

        processed = 0
        while processed != self.data_amount:
            if self.data_loader.data_queue.empty():
                continue

            data = self.data_loader.data_queue.get_nowait()
            self.data_loader.data_queue.task_done()

            self.analyse_func(data)
            self.processingUpdated.emit(data, processed)
            processed += 1

        self.processingFinished.emit()