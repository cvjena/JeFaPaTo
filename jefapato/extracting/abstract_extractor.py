__all__ = ["Extractor"]

import queue
import time

from qtpy import QtCore


class Extractor(QtCore.QThread):
    processingStarted = QtCore.Signal()
    processingUpdated = QtCore.Signal(object, object, object)
    processingPaused = QtCore.Signal()
    processingResumed = QtCore.Signal()
    processingFinished = QtCore.Signal()
    processedPercentage = QtCore.Signal(int)

    def __init__(
        self, data_queue: queue.Queue, data_amount: int, sleep_duration: float = 0.1
    ) -> None:
        super().__init__()
        self.data_queue = data_queue
        self.data_amount: int = data_amount
        self.stopped = False
        self.paused = False
        self.sleep_duration = sleep_duration

    def __del__(self):
        self.wait()

    def pause(self) -> None:
        self.paused = True
        self.processingPaused.emit()

    def resume(self) -> None:
        self.paused = False
        self.processingResumed.emit()

    def stop(self) -> None:
        self.stopped = True

    def sleep(self) -> None:
        time.sleep(self.sleep_duration)

    def toggle_pause(self) -> None:
        if self.paused:
            self.resume()
        else:
            self.pause()

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
