__all__ = ["MediapipeLandmarkExtractor"]

import queue

import cv2
import numpy as np
import structlog
import time

from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .abstract_extractor import Extractor

logger = structlog.get_logger()


class MediapipeLandmarkExtractor(Extractor):
    def __init__(self, data_queue: queue.Queue, data_amount: int) -> None:
        super().__init__(data_queue=data_queue, data_amount=data_amount)

        base_options = python.BaseOptions(model_asset_path=str(Path(__file__).parent / "models/2023-07-09_face_landmarker.task"))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_face_blendshapes=False, 
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.rect = (0, 0, 10, 10)
        self.start_time = time.time()

    def set_skip_count(self, _) -> None:
        pass

    def run(self) -> None:
        # init values
        processed = 0
        logger.info("Extractor Thread", state="starting", type="mediapipe", object=self)

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
                logger.info("Extractor Thread", state="processing", processed_p_sec=processed_p_sec)
                processed_p_sec = 0
                self.start_time = c_time

            processed_p_sec += 1
            image = self.data_queue.get()
            h, w = image.shape[:2]

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

            face_landmarker_result = self.detector.detect(mp_image)
            landmarks = np.empty((478, 3), dtype=np.int32)
            if not face_landmarker_result:
                self.processingUpdated.emit(image, self.rect, landmarks)
                processed += 1
                perc = int((processed / self.data_amount) * 100)
                self.processedPercentage.emit(perc)
                continue

            face_landmarks = face_landmarker_result.face_landmarks[0]
            for i, lm in enumerate(face_landmarks):
                landmarks[i, 0] = int(lm.x * w)
                landmarks[i, 1] = int(lm.y * h)
                landmarks[i, 2] = int(lm.z * w)

            self.rect = (
                *np.min(landmarks, axis=0)[:2],
                *np.max(landmarks, axis=0)[:2],
            )
            self.processingUpdated.emit(image, self.rect, landmarks)
            processed += 1
            perc = int((processed / self.data_amount) * 100)
            self.processedPercentage.emit(perc)

        self.processingFinished.emit()
        logger.info("Extractor Thread", state="finished", type="mediapipe", object=self)
