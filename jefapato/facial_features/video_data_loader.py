__all__ = ["VideoDataLoader"]

import queue
import time
import threading
from typing import Callable

import structlog

logger = structlog.get_logger()

class VideoDataLoader(threading.Thread):
    def __init__(self, next_item_func: Callable, queue_maxsize: int = (1 << 10)) -> None:
        super().__init__(daemon=True)

        self.start_time = time.time()
        self.next_item_func: Callable = next_item_func
        self.data_queue = queue.Queue(maxsize=queue_maxsize)
        self.data_amount = 0
        self.stopped = False

        self.processing_per_second: int = 0

    def run(self):
        logger.info("Loader Thread", state="starting", object=self)
        grabbed: bool = True

        processed_p_sec = 0

        while grabbed:
            if self.stopped:
                break
            c_time = time.time()
            if (c_time - self.start_time) > 1:
                self.start_time = c_time
                self.processing_per_second = processed_p_sec
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
