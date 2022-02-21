__all__ = ["VideoDataLoader"]

import time
from typing import Callable

import structlog

from .abstract_data_loader import DataLoader

logger = structlog.get_logger()


class VideoDataLoader(DataLoader):
    def __init__(
        self, next_item_func: Callable, queue_maxsize: int = (1 << 10)
    ) -> None:
        super().__init__(next_item_func, queue_maxsize=queue_maxsize)

    def run(self):
        grabbed: bool = True
        while grabbed:
            if self.stopped:
                break

            if not self.data_queue.full():
                (grabbed, frame) = self.next_item_func()
                if not grabbed:
                    self.stopped = True
                    break

                self.data_queue.put(frame)
            else:
                # we have to put it to sleep, else it will freeze
                # the main thread somehow
                time.sleep(0.1)

        self.stopped = True
        logger.info("Stopped loader thread.", loader=self)
