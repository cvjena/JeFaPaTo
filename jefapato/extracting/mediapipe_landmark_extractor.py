__all__ = ["MediapipeLandmarkExtractor"]

import queue

import cv2
import dlib
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
        self.skip_count = 10
        self.iter = 0
        self.scale_factor = 0.0
        self.rect = None

    def set_skip_count(self, _) -> None:
        pass

    def run(self) -> None:
        # init values
        # drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        processed = 0

        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.2,
            min_tracking_confidence=0.2,
        ) as face_mesh:
            while processed != self.data_amount and not self.stopped:
                image = self.data_queue.get()
                h, w = image.shape[:2]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                landmarks = np.zeros((478, 2), dtype=np.int32)

                # downscale image?

                results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                self.rect = dlib.rectangle(0, 0, 10, 10)

                if results.multi_face_landmarks:
                    for i, lm in enumerate(results.multi_face_landmarks[0].landmark):
                        landmarks[i, 0] = int(lm.x * w)
                        landmarks[i, 1] = int(lm.y * h)

                    self.rect = dlib.rectangle(
                        left=np.min(landmarks[:, 0]),
                        top=np.min(landmarks[:, 1]),
                        right=np.max(landmarks[:, 0]),
                        bottom=np.max(landmarks[:, 1]),
                    )
                self.processingUpdated.emit(image, self.rect, landmarks)
                processed += 1
                perc = int((processed / self.data_amount) * 100)
                self.processedPercentage.emit(perc)

        self.processingFinished.emit()
        logger.info("LandmarkExtractor finished", extractor=self)
