__all__ = ["EAR2D6", "EAR_Data", "EAR3D6", "ear_score"]

import dataclasses
from typing import Any, Dict, List

import cv2
import numpy as np
import structlog
from scipy.spatial import distance

from .abstract_feature import Feature, FeatureData

logger = structlog.get_logger()

def ear_score(eye: np.ndarray) -> float:
    """
    Compute the EAR Score for eye landmarks

    The EAR score is calculated using the formula:
    ear = (|p1 - p5| + |p2 - p4| )/ 2|p0 - p3|
    where p1, p2, p3, p4, p5, p6 are the coordinates of the 6 landmarks of the eye.
    
         p1  p2 
         |   | 
    p0 --------- p3
         |   |
         p6  p4
    
    based on:
    Soukupová and Čech 2016: Real-Time Eye Blink Detection using Facial Landmarks
    http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf

    Args:
        eye (np.ndarray): The coordinates of the 6 landmarks of the eye.

    Returns:
        float: The computed EAR score, which should be between 0 and 1
    """
    if eye is None:
        raise ValueError("eye must not be None")
    
    if not isinstance(eye, np.ndarray):
        raise TypeError(f"eye must be a numpy array, but got {type(eye)}")
    
    if eye.shape != (6, 2) and eye.shape != (6, 3): # allow for 3D landmarks
        raise ValueError(f"eye must be a 6x2 array, but got {eye.shape}")
    
    # check that no value is negative
    if np.any(eye < 0):
        # raise ValueError(f"eye must not contain negative values, but got {eye}")
        logger.warning(f"eye must not contain negative values, but got {eye}")

    
    # dont forget the 0-index
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    
    ratio = (A + B) / (2.0 * C)
    if ratio > 1.002: # allow for some rounding errors
        # raise ValueError(f"EAR score must be between 0 and 1, but got {ratio}, check your landmarks order")
        logger.warning("EAR score must be between 0 and 1, but got {ratio}, check your landmarks order")
        ratio = 1.0 
    return ratio


@dataclasses.dataclass
class EAR_Data(FeatureData):
    """A simple data class to combine the ear score results"""
    ear_l: float
    ear_r: float
    ear_valid: bool
    lm_l: np.ndarray
    lm_r: np.ndarray

class EAR(Feature):
    # index_l = np.array([362, 385, 386, 263, 374, 380])
    # index_r = np.array([ 33, 159, 158, 133, 153, 145])
    index_l = np.array([362, 385, 387, 263, 373, 380])
    index_r = np.array([ 33, 160, 158, 133, 153, 144])
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

    def draw(self, image: np.ndarray, data: EAR_Data, x_offset: int = 0, y_offset: int = 0) -> np.ndarray:        
        color_r = self.plot_info["ear_r"]["color"]
        color_l = self.plot_info["ear_l"]["color"]

        for (x, y, *_) in data.lm_l:
            cv2.circle(image, (x+x_offset, y+y_offset), 1, color_l, -1)
        for (x, y, *_) in data.lm_r:
            cv2.circle(image, (x+x_offset, y+y_offset), 1, color_r, -1)

        # draw the eye ratio
        # horizontal lines
        cv2.line(
            image,
            pt1=(data.lm_l[0, 0] + x_offset, data.lm_l[0, 1]+y_offset),
            pt2=(data.lm_l[3, 0] + x_offset, data.lm_l[3, 1]+y_offset),
            color=color_l,
            thickness=1,
        )
        cv2.line(
            image,
            pt1=(data.lm_r[0, 0]+x_offset, data.lm_r[0, 1]+y_offset),
            pt2=(data.lm_r[3, 0]+x_offset, data.lm_r[3, 1]+y_offset),
            color=color_r,
            thickness=1,
        )
        # vertical line
        t = (data.lm_l[1] + data.lm_l[2]) // 2
        b = (data.lm_l[4] + data.lm_l[5]) // 2
        cv2.line(image, (t[0]+x_offset, t[1]+y_offset), (b[0]+x_offset, b[1]+y_offset), color_l, 1)

        t = (data.lm_r[1] + data.lm_r[2]) // 2
        b = (data.lm_r[4] + data.lm_r[5]) // 2
        cv2.line(image, (t[0]+x_offset, t[1]+y_offset), (b[0]+x_offset, b[1]+y_offset), color_r, 1)
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
    def compute(self, features: np.ndarray, valid: bool) -> EAR_Data:
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

        if not valid:
            return EAR_Data(1.0, 1.0, False, lm_l, lm_r)

        ear_valid = not (np.allclose(np.zeros_like(lm_l), lm_l) and np.allclose(np.zeros_like(lm_r), lm_r))
        ear_l = ear_score(lm_l) if ear_valid else 1.0
        ear_r = ear_score(lm_r) if ear_valid else 1.0
        return EAR_Data(ear_l, ear_r, ear_valid, lm_l, lm_r)


class EAR3D6(EAR):
    plot_info: Dict[str, Dict[str, Any]] = {
        "ear_l": {"label": "EAR Left",  "color": (44, 111, 187), "width": 2}, # matte blue
        "ear_r": {"label": "EAR Right", "color": (201, 44,  17), "width": 2}, # thunderbird red
    }
    def compute(self, features: np.ndarray, valid: bool) -> EAR_Data:
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

        if not valid:
            return EAR_Data(1.0, 1.0, False, lm_l, lm_r)

        ear_valid = not (np.allclose(np.zeros_like(lm_l), lm_l) and np.allclose(np.zeros_like(lm_r), lm_r))
        ear_l = ear_score(lm_l) if ear_valid else 1.0
        ear_r = ear_score(lm_r) if ear_valid else 1.0
        return EAR_Data(ear_l, ear_r, ear_valid, lm_l, lm_r)