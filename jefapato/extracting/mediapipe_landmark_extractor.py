__all__ = ["MediapipeLandmarkExtractor"]

import queue

import cv2
import mediapipe as mp
import numpy as np
import structlog

from .abstract_extractor import Extractor

logger = structlog.get_logger()
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


class MediapipeLandmarkExtractor(Extractor):
    def __init__(self, data_queue: queue.Queue, data_amount: int) -> None:
        super().__init__(data_queue=data_queue, data_amount=data_amount)

        self.height_resize = 480
        self.rect = None

    def set_skip_count(self, _) -> None:
        pass

    def run(self) -> None:
        # init values
        processed = 0
        logger.info("Extractor Thread", state="starting", type="mediapipe", object=self)

        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.2,
            min_tracking_confidence=0.2,
        ) as face_mesh:
            while True:
                if processed == self.data_amount:
                    break

                if self.stopped:
                    break

                if self.paused:
                    self.sleep()
                    continue

                image = self.data_queue.get()
                h, w = image.shape[:2]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                landmarks = np.zeros((478, 2), dtype=np.int32)
                results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                self.rect = (0, 0, 10, 10)

                if results.multi_face_landmarks:
                    for i, lm in enumerate(results.multi_face_landmarks[0].landmark):
                        landmarks[i, 0] = int(lm.x * w)
                        landmarks[i, 1] = int(lm.y * h)

                    self.rect = (
                        np.min(landmarks[:, 0]),
                        np.min(landmarks[:, 1]),
                        np.max(landmarks[:, 0]),
                        np.max(landmarks[:, 1]),
                    )
                self.processingUpdated.emit(image, self.rect, landmarks)
                processed += 1
                perc = int((processed / self.data_amount) * 100)
                self.processedPercentage.emit(perc)

        self.processingFinished.emit()
        logger.info("Extractor Thread", state="finished", type="mediapipe", object=self)
