__all__ = ["VideoDataLoader"]

import queue
import time
import threading
from typing import Callable
import cv2
import numpy as np

import structlog

from .queue_items import InputQueueItem

logger = structlog.get_logger()

class VideoDataLoader(threading.Thread):
    def __init__(
        self, 
        next_item_func: Callable, 
        data_amount: int | None = None,
        queue_maxsize: int = (1 << 10)
    ) -> None:
        super().__init__(daemon=True)
        assert data_amount is not None, "data_amount must be set"

        self.start_time = time.time()
        self.next_item_func: Callable = next_item_func
        self.data_queue: queue.Queue[InputQueueItem] = queue.Queue(maxsize=queue_maxsize)
        self.stopped = False
        self.data_amount = data_amount

        self.processing_per_second: int = 0
        

    def run(self):
        logger.info("Loader Thread", state="starting", object=self)
        grabbed: bool = True

        processed_p_sec = 0
        frame_id = 0
        
        w, h = None, None
        
        while True:
            if self.stopped:
                break
            c_time = time.time()
            if (c_time - self.start_time) > 1:
                self.start_time = c_time
                self.processing_per_second = processed_p_sec
                processed_p_sec = 0
            
            if not self.data_queue.full():
                frame_id += 1
                (grabbed, frame) = self.next_item_func()
                if frame is not None and w is None and h is None:
                    h, w = frame.shape[:2]
                # if we failed to grab but more data is available, we can
                # assume that only the current frame is broken and we can
                # continue 
                if not grabbed and frame_id < self.data_amount:
                    # TODO: log this
                    logger.warning("Failed to grab frame", frame_id=frame_id)
                    frame = np.zeros((h, w, 3), dtype=np.uint8)
                # this is the last frame, we can stop
                elif not grabbed:
                    self.stopped = True
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.data_queue.put(InputQueueItem(frame=frame, timestamp=c_time))
                processed_p_sec += 1
            else:
                # we have to put it to sleep, else it will freeze
                # the main thread somehow
                time.sleep(0.1)

        logger.info("Loader Thread", state="finished")
