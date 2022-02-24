__all__ = ["EARFeature", "EARData"]

import dataclasses
from typing import Any, Dict, List

import cv2
import numpy as np
import structlog
from scipy.spatial import distance

from .abstract_feature import Feature, FeatureData

logger = structlog.get_logger()


@dataclasses.dataclass
class EARData(FeatureData):
    """A simple data class to combine the ear score results"""

    lm_l: np.ndarray
    lm_r: np.ndarray
    ear_l: float
    ear_r: float
    ear_valid: bool

    def as_row(self) -> List[str]:
        """Return the data as a row as in the same order as the header"""
        dlib_l_x = self.lm_l[:, 0]
        dlib_l_y = self.lm_l[:, 1]
        dlib_r_x = self.lm_r[:, 0]
        dlib_r_y = self.lm_r[:, 1]
        return (
            list(map(str, dlib_l_x))
            + list(map(str, dlib_l_y))
            + list(map(str, dlib_r_x))
            + list(map(str, dlib_r_y))
            + [f"{self.ear_l: .8f}", f"{self.ear_r: .8f}", str(self.ear_valid)]
        )

    def draw(self, image: np.ndarray) -> np.ndarray:
        # the dlib landmark points are in the format (x, y)
        for (x, y) in self.lm_l:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        for (x, y) in self.lm_r:
            cv2.circle(image, (x, y), 1, (255, 0, 0), -1)

        # draw the eye ratio
        # horizontal lines
        cv2.line(
            image,
            pt1=(self.lm_l[0, 0], self.lm_l[0, 1]),
            pt2=(self.lm_l[3, 0], self.lm_l[3, 1]),
            color=(0, 0, 255),
            thickness=1,
            # lineType=cv2.LINE_AA,
        )
        cv2.line(
            image,
            pt1=(self.lm_r[0, 0], self.lm_r[0, 1]),
            pt2=(self.lm_r[3, 0], self.lm_r[3, 1]),
            color=(255, 0, 0),
            thickness=1,
            # lineType=cv2.LINE_AA,
        )
        # vertical line
        t = (self.lm_l[1] + self.lm_l[2]) // 2
        b = (self.lm_l[4] + self.lm_l[5]) // 2
        cv2.line(image, (t[0], t[1]), (b[0], b[1]), (0, 0, 255), 1)

        t = (self.lm_r[1] + self.lm_r[2]) // 2
        b = (self.lm_r[4] + self.lm_r[5]) // 2
        cv2.line(image, (t[0], t[1]), (b[0], b[1]), (255, 0, 0), 1)

        return image


class EARFeature(Feature):
    """EyeBlinking Classifier Class for 68 facial landmarks features

    This class implements a classifier specificly for the 68 landmarking
    schemata.
    """

    # the dict keys need to be the same name as the class attributes in the
    # dataclass in self.d_type
    # because we access with with getattr(self, key)
    plot_info: Dict[str, Dict[str, Any]] = {
        "ear_l": {"label": "EAR Left", "color": "b", "width": 2},
        "ear_r": {"label": "EAR Right", "color": "r", "width": 2},
    }

    def __init__(self, backend: str = "dlib") -> None:
        super().__init__()
        self.d_type = EARData
        self.backend = backend

        if self.backend == "dlib":
            self.index_eye_l: slice = slice(42, 48)
            self.index_eye_r: slice = slice(36, 42)
        else:
            logger.error("Only dlib backend is supported in EARFeature")

    def ear_score(self, eye: np.ndarray) -> float:
        """Compute the EAR Score for eye landmarks

        Soukupová and Čech 2016: Real-Time Eye Blink Detection using Facial Landmarks
        http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf

        ear = (|p2 - p6| + |p3 - p5| )/ 2|p1-p4|
                   A           B           C

        Args:
            eye (np.ndarray): the coordinates of the 6 landmarks

        Returns:
            float: computed EAR score
        """

        # dont forget the 0-index
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])

        return (A + B) / (2.0 * C)

    def compute(self, in_data: np.ndarray) -> EARData:
        """
        Compute the feature for the given data

        Parameters
        ----------
        in_data : np.ndarray
            The data to compute the feature for the EAR score
            If the in_data is full of zeros then the landmarks failed to be computed
            and the EAR score is set to 1.0 and marked as invalid

        Returns
        -------
        EARData
            The computed features with the raw data
        """
        if self.backend != "dlib":
            return EARData(np.zeros((5, 2)), np.zeros((5, 2)), 1.0, 1.0, False)

        # extract the eye landmarks
        lm_l = in_data[self.index_eye_l]
        lm_r = in_data[self.index_eye_r]
        ear_valid = not (
            np.allclose(np.zeros_like(lm_l), lm_l)
            and np.allclose(np.zeros_like(lm_r), lm_r)
        )
        ear_l = self.ear_score(lm_l) if ear_valid else 1.0
        ear_r = self.ear_score(lm_r) if ear_valid else 1.0
        return EARData(lm_l, lm_r, ear_l, ear_r, ear_valid)

    def get_header(self) -> List[str]:
        """
        Returns
        -------
        List[str]
            The header for the feature including the valid and all the landmarks
        """
        if self.backend != "dlib":
            return [""] * 20 + ["EAR Left", "EAR Right", "EAR Valid"]

        dlib_l_x = [
            f"dlib_l_x_{i:02d}"
            for i in range(self.index_eye_l.start, self.index_eye_l.stop)
        ]
        dlib_l_y = [
            f"dlib_l_y_{i:02d}"
            for i in range(self.index_eye_l.start, self.index_eye_l.stop)
        ]
        dlib_r_x = [
            f"dlib_r_x_{i:02d}"
            for i in range(self.index_eye_r.start, self.index_eye_r.stop)
        ]
        dlib_r_y = [
            f"dlib_r_y_{i:02d}"
            for i in range(self.index_eye_r.start, self.index_eye_r.stop)
        ]
        return (
            dlib_l_x
            + dlib_l_y
            + dlib_r_x
            + dlib_r_y
            + ["dlib_ear_l", "dlib_ear_r", "dlib_ear_valid"]
        )
