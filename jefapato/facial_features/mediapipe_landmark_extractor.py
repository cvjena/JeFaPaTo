__all__ = ["MediapipeLandmarkExtractor"]

import queue
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import structlog
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from qtpy.QtCore import QThread, Signal

from .queue_items import InputQueueItem, AnalyzeQueueItem

logger = structlog.get_logger()


class Extractor(QThread):
    processingStarted = Signal()
    processingUpdated = Signal(object)
    processingPaused = Signal()
    processingResumed = Signal()
    processingFinished = Signal()
    processedPercentage = Signal(int)

    def __init__(
        self, data_queue: queue.Queue[InputQueueItem], data_amount: int, sleep_duration: float = 0.1
    ) -> None:
        super().__init__()
        self.data_queue = data_queue
        self.data_amount: int = int(data_amount) - 1 
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

    def set_skip_count(self, _) -> None:
        pass

    def run(self) -> None:
        # init values
        processed = 0
        logger.info("Extractor Thread", state="starting", data_amount=self.data_amount)

        # wait for the queue to be filled
        time.sleep(1)

        processed_p_sec = 0
        while True:
            if processed == self.data_amount:
                break

            if self.stopped:
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
                # TODO here we might have the edge case of processing being faster than the queue being filled
                #      we should rather count how often we sleep after each other and then break if we sleep too often
                logger.info("Extractor Thread", state="QUEUE EMPTY", data_amount=self.data_amount, processed=processed)
                break

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
            self.processingUpdated.emit(item)
            processed += 1
            perc = int((processed / self.data_amount) * 100)
            self.processedPercentage.emit(perc)

        self.processedPercentage.emit(100)
        self.processingFinished.emit()
        logger.info("Extractor Thread", state="finished")
