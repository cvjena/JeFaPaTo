from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import cv2
import dlib
import numpy as np
from imutils import face_utils


def scale_bbox(bbox: dlib.rectangle, scale: float) -> dlib.rectangle:
    """Scale and pad a dlib.rectangle

    Copyright: Tim Büchner tim.buechner@uni-jena.de

    This function will scale a dlib.rectangle and can also apply additional padding
    to the rectangle. If a padding value of -1 is given, the padding is will be
    a fourth of the respective widht and height.

    Args:
        bbox (dlib.rectangle): rectangle which will be scaled and optionally padded
        scale (float): scaling factor for the rectangle
        padding (int, optional):    Additional padding for the rectangle. Defaults to 0.
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


class FeatureExtractor(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def extract_features(self, data: Any) -> Any:
        """Abstract class for feature extraction

        Args:
            data (Any): data from which the features should be extracted

        Returns:
            Any: Specific feature description
        """
        pass


class LandMarkFeatureExtractor(FeatureExtractor):
    def __init__(self) -> None:
        super().__init__()
        self.shape_predictor_file = "./data/shape_predictor_68_face_landmarks.dat"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.shape_predictor_file)

        self.height_resize = 480
        self.skip_count = 10
        self.rects: List[dlib.rectangle] = list()
        self.iter = 0
        self.scale_factor = 0.0

    def set_skip(self, value: int) -> None:
        self.skip_count = value

    def extract_features(
        self, data: np.ndarray
    ) -> List[Tuple[dlib.rectangle, np.ndarray]]:
        """Extract landmark features from images with faces

        Extract facial features from images, if faces are present.
        If not an empty list will be returned.

        Args:
            data (np.ndarray): rgb image which could contain faces

        Returns:
            List[Tuple[dlib.rectangle, np.ndarray]]: A list of tuple which describe
                the bounding box including the 68 facial features
        """
        if self.iter % self.skip_count == 0:
            height = data.shape[0]
            self.scale_factor = 1 / (float(height) / self.height_resize)

            data_sc = cv2.resize(
                data,
                None,
                fx=self.scale_factor,
                fy=self.scale_factor,
                interpolation=cv2.INTER_LINEAR,
            )
            self.rects = self.detector(data_sc, 0)

        results: List[Tuple[dlib.rectangle, np.ndarray]] = []

        for _, rect in enumerate(self.rects):
            rect = scale_bbox(rect, 1 / self.scale_factor)
            shape = self.predictor(data, rect)
            shape = face_utils.shape_to_np(shape)
            results.append((rect, shape))

        self.iter += 1
        return results
