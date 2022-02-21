__all__ = ["DataLoader"]

import abc
import queue
import threading
from typing import Callable


class DataLoader(abc.ABC, threading.Thread):
    def __init__(
        self, next_item_func: Callable, queue_maxsize: int = (1 << 10)
    ) -> None:
        super().__init__(daemon=True)
        self.next_item_func: Callable = next_item_func
        self.data_queue = queue.Queue(maxsize=queue_maxsize)
        self.data_amount = 0
        self.stopped = False

    @abc.abstractmethod
    def run(self):
        pass
