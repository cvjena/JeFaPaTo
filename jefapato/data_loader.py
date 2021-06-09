from abc import ABC, abstractmethod
from threading import Thread
from typing import Callable

from pathlib import Path
from queue import Queue

import cv2

class DataLoader(ABC, Thread):

    def __init__(self, next_item_func: Callable, queue_maxsize: int=0) -> None:
        super().__init__(daemon=True)
        self.next_item_func: Callable = next_item_func
        self.data_queue: Queue = Queue(maxsize=queue_maxsize)
        self.data_amount: int = 0
        self.stopped: bool = False

    @abstractmethod
    def run(self):
        pass

class VideoDataLoader(DataLoader):
    def __init__(self, next_item_func: Callable, queue_maxsize: int=0) -> None:
        super().__init__(next_item_func, queue_maxsize=queue_maxsize)

    def run(self):
        # TODO add logging
        grabbed: bool = True
        while grabbed:
            if self.stopped:
                return
            
            if not self.data_queue.full():
                (grabbed, frame) = self.next_item_func()
                if not grabbed:
                    self.stopped = True
                    return

                self.data_queue.put(frame)

        self.stopped = True
