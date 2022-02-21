__all__ = ["Extractor"]

import queue

import numpy as np
from qtpy import QtCore


class Extractor(QtCore.QThread):
    processingStarted = QtCore.Signal()
    processingUpdated = QtCore.Signal(np.ndarray, np.ndarray)
    processingFinished = QtCore.Signal()
    processedPercentage = QtCore.Signal(int)

    def __init__(self, data_queue: queue.Queue, data_amount: int) -> None:
        super().__init__()
        self.data_queue = data_queue
        self.data_amount: int = data_amount
        self.stopped = False

    def __del__(self):
        self.wait()

    def run(self):
        raise NotImplementedError(
            "Extractor.run() must be implemented in the inherited class."
        )

    # def run(self):
    #     self.processingStarted.emit()

    #     processed = 0
    #     while processed != self.data_amount and not self.stopped:
    #         data = self.data_loader.data_queue.get()

    #         self.analyse_func(data)
    #         self.processingUpdated.emit(data, processed)
    #         processed += 1

    #         perc = int((processed / self.data_amount) * 100)
    #         self.processedPercentage.emit(perc)

    #     self.processingFinished.emit()
