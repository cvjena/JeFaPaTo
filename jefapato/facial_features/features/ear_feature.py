__all__ = ["EAR2D6_Feature", "EAR2D6_Data"]

import dataclasses
from typing import Any, Dict, List

import cv2
import numpy as np
import structlog
from scipy.spatial import distance

from .abstract_feature import Feature, FeatureData

logger = structlog.get_logger()


@dataclasses.dataclass
class EAR2D6_Data(FeatureData):
    """A simple data class to combine the ear score results"""

    lm_l: np.ndarray
    lm_r: np.ndarray
    ear_l: float
    ear_r: float
    ear_valid: bool

    def as_row(self) -> List[str]:
        """Return the data as a row as in the same order as the header"""
        return (
            list(map(str, self.lm_l[:, 0]))
            + list(map(str, self.lm_l[:, 1]))
            + list(map(str, self.lm_r[:, 0]))
            + list(map(str, self.lm_r[:, 1]))
            + [f"{self.ear_l: .8f}", f"{self.ear_r: .8f}", str(self.ear_valid)]
        )

    def draw(self, image: np.ndarray) -> np.ndarray:
        # the dlib landmark points are in the format (x, y)
        for x, y, _ in self.lm_l:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        for x, y, _ in self.lm_r:
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


class EAR2D6_Feature(Feature):
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
    mp_eye_l = np.array([362, 385, 386, 263, 374, 380])
    mp_eye_r = np.array([33, 159, 158, 133, 153, 145])

    def __init__(self) -> None:
        super().__init__()
        self.d_type = EAR2D6_Data
        self.index_l = self.mp_eye_l
        self.index_r = self.mp_eye_r

    def ear_score(self, eye: np.ndarray) -> float:
        """Compute the EAR Score for eye landmarks

        Soukupová and Čech 2016: Real-Time Eye Blink Detection using Facial Landmarks
        http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf

        ear = (|p2 - p6| + |p3 - p5| )/ 2|p1 - p4|
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

    def compute(self, in_data: np.ndarray) -> EAR2D6_Data:
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
        # extract the eye landmarks
        lm_l = in_data[self.index_l]
        lm_r = in_data[self.index_r]
        ear_valid = not (np.allclose(np.zeros_like(lm_l), lm_l) and np.allclose(np.zeros_like(lm_r), lm_r))
        ear_l = self.ear_score(lm_l) if ear_valid else 1.0
        ear_r = self.ear_score(lm_r) if ear_valid else 1.0
        return EAR2D6_Data(lm_l, lm_r, ear_l, ear_r, ear_valid)

    def get_header(self) -> List[str]:
        """
        Returns
        -------
        List[str]
            The header for the feature including the valid and all the landmarks
        """
        l_x = [f"l_x_{i:03d}" for i in self.index_l]
        l_y = [f"l_y_{i:03d}" for i in self.index_l]
        r_x = [f"r_x_{i:03d}" for i in self.index_r]
        r_y = [f"r_y_{i:03d}" for i in self.index_r]
        return l_x + l_y + r_x + r_y + ["ear_l", "ear_r", "ear_valid"]
