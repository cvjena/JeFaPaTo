__all__ = ["VideoDataLoader"]

import queue
import threading
import time
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
        queue_maxsize: int = (1 << 10),  # 1024 items
        rotation: str = "None",
    ) -> None:
        """
        Initializes the VideoDataLoader.

        Args:
            next_item_func (Callable): A function that returns the next item to be processed.
            data_amount (int | None, optional): The total amount of data to be processed. Must be set to a non-None value.
            queue_maxsize (int, optional): The maximum size of the queue. Defaults to 1024 items.
            rotation (str, optional): The rotation to be applied to the data. Can be "None", "90", or "-90". Defaults to "None".

        Raises:
            AssertionError: If data_amount is None.
        """
        super().__init__(daemon=True)
        assert data_amount is not None, "data_amount must be set"
        assert rotation in ["None", "90", "-90"], "rotation must be 'None', '90', or '-90'"

        self.start_time = time.time()
        self.next_item_func: Callable = next_item_func
        self.data_queue: queue.Queue[InputQueueItem] = queue.Queue(maxsize=queue_maxsize)
        self.stopped = False
        self.data_amount = data_amount
        self.processing_per_second: float = 0

        if rotation == "None":
            self.rotation_fc = lambda x: x
        elif rotation == "90":
            self.rotation_fc = lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == "-90":
            self.rotation_fc = lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def run(self) -> None:
        """
        Starts the thread.
        """

        logger.info("Loader Thread", state="starting", object=self)
        grabbed: bool = True

        processed_p_sec = 0
        frame_id = 0

        w: int = 256
        h: int = 256

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
                # assume that only the current frame is broken and we can continue
                if not grabbed and frame_id < self.data_amount:
                    logger.warning("Failed to grab frame", frame_id=frame_id)
                    frame = np.zeros((h, w, 3), dtype=np.uint8)
                # this is the last frame, we can stop
                elif not grabbed:
                    self.stopped = True
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.rotation_fc(frame)
                self.data_queue.put(InputQueueItem(frame=frame, timestamp=c_time))
                processed_p_sec += 1
            else:
                # we have to put it to sleep, else it will freeze
                # the main thread somehow
                time.sleep(0.1)

        logger.info("Loader Thread", state="finished")
