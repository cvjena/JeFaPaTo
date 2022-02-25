__all__ = ["DlibLandmarkExtractor"]

import pathlib
import queue

import cv2
import dlib
import numpy as np
import structlog

from .abstract_extractor import Extractor

logger = structlog.get_logger()


def scale_bbox(bbox: dlib.rectangle, scale: float) -> dlib.rectangle:
    """Scale and pad a dlib.rectangle

    Copyright: Tim Büchner tim.buechner@uni-jena.de

    This function will scale a dlib.rectangle and can also apply additional padding
    to the rectangle. If a padding value of -1 is given, the padding is will be
    a fourth of the respective widht and height.

    Args:
        bbox (dlib.rectangle): rectangle which will be scaled and optionally padded
        scale (float): scaling factor for the rectangle
        padding (int, optional):    Additional padding for the rectangle. (defaults 0)
                                    If -1 use fourth of width and height for padding.

    Returns:
        dlib.rectangle: newly scaled (and optionally padded) dlib.rectangle
    """
    left = int(bbox.left() * scale)
    top = int(bbox.top() * scale)
    right = int(bbox.right() * scale)
    bottom = int(bbox.bottom() * scale)

    return dlib.rectangle(
        left=left,
        top=top,
        right=right,
        bottom=bottom,
    )


class DlibLandmarkExtractor(Extractor):
    def __init__(self, data_queue: queue.Queue, data_amount: int) -> None:
        super().__init__(data_queue=data_queue, data_amount=data_amount)

        self.shape_predictor_file = self.__get_shape_predictor()
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.shape_predictor_file)

        self.height_resize = 480
        self.skip_count = 10
        self.iter = 0
        self.scale_factor = 0.0
        self.rect = None

    def __get_shape_predictor(self) -> pathlib.Path:
        static_path = pathlib.Path(__file__).parent.parent.parent / "__static__"
        file_path = static_path / "shape_predictor_68_face_landmarks.dat"
        if file_path.exists() and file_path.is_file():
            return file_path.as_posix()
        else:
            raise FileNotFoundError("shape_predictor_68_face_landmarks.dat not found")

    def set_skip_count(self, skip_count: int) -> None:
        self.skip_count = skip_count

    def run(self) -> None:
        self.processingStarted.emit()

        processed = 0
        while processed != self.data_amount and not self.stopped:
            if self.paused:
                self.sleep()
                continue

            image = self.data_queue.get()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if processed % self.skip_count == 0:
                height = image.shape[0]
                self.scale_factor = 1 / (float(height) / self.height_resize)

                data_sc = cv2.resize(
                    image, (0, 0), fx=self.scale_factor, fy=self.scale_factor
                )
                rects = self.detector(data_sc, 0)

                # we didnt find any faces
                if len(rects) == 0:
                    self.rect = None
                else:
                    self.rect = rects[0]

            landmarks = np.zeros((68, 2), dtype=np.int32)

            if self.rect is not None:
                rect = scale_bbox(self.rect, 1 / self.scale_factor)
                shape = self.predictor(image, rect)

                for i in range(0, 68):
                    landmarks[i, 0] = shape.part(i).x
                    landmarks[i, 1] = shape.part(i).y

            self.processingUpdated.emit(image, rect, landmarks)
            processed += 1
            perc = int((processed / self.data_amount) * 100)
            self.processedPercentage.emit(perc)

        self.processingFinished.emit()
        logger.info("LandmarkExtractor finished", extractor=self)
