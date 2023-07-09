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

        self.start_time = time.time()

    def run(self):
        logger.info("Loader Thread", state="starting", object=self)
        grabbed: bool = True

        processed_p_sec = 0

        while grabbed:
            if self.stopped:
                break

            c_time = time.time()
            if (c_time - self.start_time) > 1:
                logger.info("Loader Thread", state="processing", processed_p_sec=processed_p_sec, queue_size=self.data_queue.qsize())
                self.start_time = c_time
                processed_p_sec = 0 

            if not self.data_queue.full():
                (grabbed, frame) = self.next_item_func()
                if not grabbed:
                    self.stopped = True
                    break

                self.data_queue.put(frame)
                processed_p_sec += 1
            else:
                # we have to put it to sleep, else it will freeze
                # the main thread somehow
                time.sleep(0.1)

        logger.info("Loader Thread", state="finished", object=self)
