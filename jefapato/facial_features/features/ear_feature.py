__all__ = ["EAR2D6", "EAR_Data", "EAR3D6"]

import dataclasses
from typing import Any, Dict, List

import cv2
import numpy as np
import structlog
from scipy.spatial import distance

from .abstract_feature import Feature, FeatureData

logger = structlog.get_logger()

def ear_score(eye: np.ndarray) -> float:
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


@dataclasses.dataclass
class EAR_Data(FeatureData):
    """A simple data class to combine the ear score results"""
    ear_l: float
    ear_r: float
    ear_valid: bool
    lm_l: np.ndarray
    lm_r: np.ndarray

class EAR(Feature):
    index_l = np.array([362, 385, 386, 263, 374, 380])
    index_r = np.array([ 33, 159, 158, 133, 153, 145])
    def get_header(self) -> List[str]:
        """
        Returns
        -------
        List[str]
            The header for the feature including the valid and all the landmarks
        """
        return [f"{self.__class__.__name__}_l", f"{self.__class__.__name__}_r", f"{self.__class__.__name__}_valid"]

    def as_row(self, data: EAR_Data) -> List[str]:
        """Return the data as a row as in the same order as the header"""
        return [f"{data.ear_l: .8f}", f"{data.ear_r: .8f}", str(data.ear_valid)]

    def draw(self, image: np.ndarray, data: EAR_Data) -> np.ndarray:        
        color_r = self.plot_info["ear_r"]["color"]
        color_l = self.plot_info["ear_l"]["color"]

        for (x, y, *_) in data.lm_l:
            cv2.circle(image, (x, y), 1, color_l, -1)
        for (x, y, *_) in data.lm_r:
            cv2.circle(image, (x, y), 1, color_r, -1)

        # draw the eye ratio
        # horizontal lines
        cv2.line(
            image,
            pt1=(data.lm_l[0, 0], data.lm_l[0, 1]),
            pt2=(data.lm_l[3, 0], data.lm_l[3, 1]),
            color=color_l,
            thickness=1,
        )
        cv2.line(
            image,
            pt1=(data.lm_r[0, 0], data.lm_r[0, 1]),
            pt2=(data.lm_r[3, 0], data.lm_r[3, 1]),
            color=color_r,
            thickness=1,
        )
        # vertical line
        t = (data.lm_l[1] + data.lm_l[2]) // 2
        b = (data.lm_l[4] + data.lm_l[5]) // 2
        cv2.line(image, (t[0], t[1]), (b[0], b[1]), color_l, 1)

        t = (data.lm_r[1] + data.lm_r[2]) // 2
        b = (data.lm_r[4] + data.lm_r[5]) // 2
        cv2.line(image, (t[0], t[1]), (b[0], b[1]), color_r, 1)
        return image

class EAR2D6(EAR):
    """Eye-Aspect-Ratio Feature based on 2D landmarks

    This class implements a classifier specificly for the 68 landmarking
    schemata.
    """

    plot_info: dict[str, dict[str, Any]] = {
        "ear_l": {"label": "EAR Left",  "color": (  0,   0, 255), "width": 2},
        "ear_r": {"label": "EAR Right", "color": (255,   0,   0), "width": 2},
    }
    def compute(self, features: np.ndarray) -> EAR_Data:
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
        lm_l = features[self.index_l, :2]
        lm_r = features[self.index_r, :2]

        ear_valid = not (np.allclose(np.zeros_like(lm_l), lm_l) and np.allclose(np.zeros_like(lm_r), lm_r))
        ear_l = ear_score(lm_l) if ear_valid else 1.0
        ear_r = ear_score(lm_r) if ear_valid else 1.0
        return EAR_Data(ear_l, ear_r, ear_valid, lm_l, lm_r)


class EAR3D6(EAR):
    plot_info: Dict[str, Dict[str, Any]] = {
        "ear_l": {"label": "EAR Left",  "color": (44, 111, 187), "width": 2}, # matte blue
        "ear_r": {"label": "EAR Right", "color": (201, 44,  17), "width": 2}, # thunderbird red
    }
    def compute(self, features: np.ndarray) -> EAR_Data:
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
        lm_l = features[self.index_l]
        lm_r = features[self.index_r]

        ear_valid = not (np.allclose(np.zeros_like(lm_l), lm_l) and np.allclose(np.zeros_like(lm_r), lm_r))
        ear_l = ear_score(lm_l) if ear_valid else 1.0
        ear_r = ear_score(lm_r) if ear_valid else 1.0
        return EAR_Data(ear_l, ear_r, ear_valid, lm_l, lm_r)