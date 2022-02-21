__all__ = ["VideoDataLoader"]

import time
from typing import Callable

from .abstract_data_loader import DataLoader


class VideoDataLoader(DataLoader):
    def __init__(
        self, next_item_func: Callable, queue_maxsize: int = (1 << 10)
    ) -> None:
        super().__init__(next_item_func, queue_maxsize=queue_maxsize)

    def run(self):
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
            else:
                # we have to put it to sleep, else it will freeze
                # the main thread somehow
                time.sleep(0.1)

        self.stopped = True
