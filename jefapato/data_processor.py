from typing import Callable

from .data_loader import DataLoader

import numpy as np

from PyQt5.QtCore import QThread, pyqtSignal


class DataProcessor(QThread):
    processingStarted: pyqtSignal = pyqtSignal()
    processingUpdated: pyqtSignal = pyqtSignal(np.ndarray, int)
    processingFinished: pyqtSignal = pyqtSignal()
    processedPercentage: pyqtSignal = pyqtSignal(int)

    def __init__(
        self, analyse_func: Callable, data_amount: int, data_loader: DataLoader
    ) -> None:
        super().__init__()
        self.analyse_func: Callable = analyse_func
        self.data_loader: DataLoader = data_loader
        self.data_amount: int = data_amount
        self.stopped = False

    def __del__(self):
        self.wait()

    def run(self):
        # TODO add loggin
        self.processingStarted.emit()

        processed = 0
        while processed != self.data_amount and not self.stopped:
            data = self.data_loader.data_queue.get()

            self.analyse_func(data)
            self.processingUpdated.emit(data, processed)
            processed += 1

            perc = int((processed / self.data_amount) * 100)
            self.processedPercentage.emit(perc)

        self.processingFinished.emit()
