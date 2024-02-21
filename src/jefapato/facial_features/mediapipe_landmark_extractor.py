__all__ = ["MediapipeLandmarkExtractor"]

import queue
import time
from pathlib import Path

import mediapipe as mp
import numpy as np
import structlog
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from threading import Thread
from .queue_items import AnalyzeQueueItem, InputQueueItem
import pluggy

logger = structlog.get_logger()


class Extractor(Thread):
    hookimpl = pluggy.HookimplMarker("Extractor")
    hookspec = pluggy.HookspecMarker("Extractor")
    
    def __init__(
        self, data_queue: queue.Queue[InputQueueItem], data_amount: int, sleep_duration: float = 0.08
    ) -> None:
        super().__init__()
        self.data_queue = data_queue
        self.data_amount: int = int(data_amount) - 1 
        self.stopped = False
        self.paused = False
        self.sleep_duration = sleep_duration
        self.processing_per_second: int = 0
        
        self.pm = pluggy.PluginManager("Extractor")

    def register(self, object) -> None:
        self.pm.register(object)
 
    @hookspec
    def handle_update(self, item: AnalyzeQueueItem) -> None:
        """
        A trigger to be implemented by the hook to handle the according event.
        """
    
    @hookspec
    def handle_finished(self) -> None:
        """
        A trigger to be implemented by the hook to handle the according event.
        """
    
    @hookspec
    def update_progress(self, perc: int) -> None:
        """
        A trigger to be implemented by the hook to handle the according event.
        """

    @hookspec
    def handle_pause(self) -> None:
        """
        A trigger to be implemented by the hook to handle the according event.
        """
    
    @hookspec
    def handle_resume(self) -> None:
        """
        A trigger to be implemented by the hook to handle the according event.
        """

    def pause(self) -> None:
        self.paused = True
        self.pm.hook.handle_pause()

    def resume(self) -> None:
        self.paused = False
        self.pm.hook.handle_resume()

    def stop(self) -> None:
        self.stopped = True

    def sleep(self) -> None:
        time.sleep(self.sleep_duration)

    def toggle_pause(self) -> None:
        if self.paused:
            self.resume()
        else:
            self.pause()

    def isRunning(self) -> bool:
        """
        Check if the mediapipe landmark extractor is running.

        NOTE: This is not the same as the thread state. The thread can be alive
        but the extractor can be paused. However, this is a workaround to be compatible
        with the old QThread implementation.

        Returns:
            bool: True if the extractor is running, False otherwise.
        """
        # check if the thread is alive
        return self.is_alive() and not self.paused

class MediapipeLandmarkExtractor(Extractor):
    def __init__(
        self, 
        data_queue: queue.Queue[InputQueueItem], 
        data_amount: int,
        bbox_slice: tuple[int, int, int, int] | None = None,
    ) -> None:
        super().__init__(data_queue=data_queue, data_amount=data_amount)

        base_options = python.BaseOptions(model_asset_path=str(Path(__file__).parent / "models/2023-07-09_face_landmarker.task"))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_face_blendshapes=True, 
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.start_time = time.time()
        self.processing_per_second: int = 0
        self.bbox_slice = bbox_slice
        self.processed = 0

    def run(self) -> None:
        # init values
        self.processed = 0
        logger.info("Extractor Thread", state="starting", data_amount=self.data_amount)

        # wait for the queue to be filled
        self.sleep()

        empty_in_a_row = 0
        processed_p_sec = 0

        while True:
            if self.processed == self.data_amount or self.stopped:
                break

            if self.paused:
                self.sleep()
                continue

            # check if 1 second has passed
            c_time = time.time()
            if (c_time - self.start_time) > 1:
                self.start_time = c_time
                self.processing_per_second = processed_p_sec
                processed_p_sec = 0

            processed_p_sec += 1
            if self.data_queue.empty():
                empty_in_a_row += 1
                self.sleep()
                if empty_in_a_row > 20:
                    logger.info("Extractor Thread", state="Queue Emptpy", data_amount=self.data_amount, processed=self.processed)
                    self.stopped = True
                continue
            empty_in_a_row = 0

            frame = self.data_queue.get().frame
            image = frame.copy()

            if self.bbox_slice:
                y1, y2, x1, x2 = self.bbox_slice
                image = image[y1:y2, x1:x2].copy()

            h, w = image.shape[:2]
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

            face_landmarker_result = self.detector.detect(mp_image)
            landmarks = np.empty((478, 3), dtype=np.int32)
            blendshapes = {}

            valid = False
            if face_landmarker_result.face_landmarks:
                valid = True
                face_landmarks = face_landmarker_result.face_landmarks[0]
                for i, lm in enumerate(face_landmarks):
                    landmarks[i, 0] = int(lm.x * w)
                    landmarks[i, 1] = int(lm.y * h)
                    landmarks[i, 2] = int(lm.z * w)

                for face_blendshape in face_landmarker_result.face_blendshapes[0]:
                    blendshapes[face_blendshape.category_name] = face_blendshape.score

            x_offset = 0 if self.bbox_slice is None else self.bbox_slice[2]
            y_offset = 0 if self.bbox_slice is None else self.bbox_slice[0]

            item = AnalyzeQueueItem(frame, valid, landmarks, blendshapes, x_offset, y_offset)
            
            self.pm.hook.handle_update(item=item)
            
            self.processed += 1
            perc = int((self.processed / self.data_amount) * 100)
            
            self.pm.hook.update_progress(perc=perc)

        self.pm.hook.update_progress(perc=100.0)
        self.pm.hook.handle_finished()
        logger.info("Extractor Thread", state="finished")
