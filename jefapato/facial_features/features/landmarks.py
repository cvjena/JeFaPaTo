__all__ = ["Landmarks478", "Landmarks68"]

import dataclasses

import cv2
import numpy as np

from .abstract_feature import Feature, FeatureData


@dataclasses.dataclass
class LandmarksData(FeatureData):
    """A simple data class to combine the ear score results"""
    landmarks: np.ndarray # the 478 landmarks
    lm_valid: bool # if the landmarks are valid

class Landmarks478(Feature):
    def get_header(self) -> list[str]:
        """
        Returns
        -------
        List[str]
            The header for the feature including the valid and all the landmarks
        """
        return [f"{self.__class__.__name__}_valid"] + [f"{self.__class__.__name__}_{i}" for i in range(478)]

    def as_row(self, data: LandmarksData) -> list[str]:
        """Return the data as a row as in the same order as the header"""

        # make the landmarks a list of strings
        return [str(data.lm_valid)] + [f"({i[0]}, {i[1]}, {i[2]})" for i in data.landmarks]

    def draw(self, image: np.ndarray, data: LandmarksData, x_offset: int = 0, y_offset: int = 0) -> np.ndarray:
        """Draw the landmarks on the image"""
        color = (0, 0 , 0)
        for (x, y, *_) in data.landmarks:
            cv2.circle(image, (x+x_offset, y+y_offset), 2, color, -1)
        return image


    def compute(self, features: np.ndarray, valid: bool) -> LandmarksData:
        """Compute the 478 landmarks for the image"""
        return LandmarksData(landmarks=features, lm_valid=valid)
    

class Landmarks68(Feature):
    """
    The input is the 478 landmarks from mediapipe (BlazeModel)
    but we only use the 68 landmarks from the 478 landmarks
    which are the same as the 68 landmarks from dlib
    """

    # https://stackoverflow.com/a/71501885
    lms_478_to_68 = [
        162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
        296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
        380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87
    ]

    def get_header(self) -> list[str]:
        """
        Returns
        -------
        List[str]
            The header for the feature including the valid and all the landmarks
        """
        return [f"{self.__class__.__name__}_valid"] + [f"{self.__class__.__name__}_{i}" for i in range(68)]
    
    def as_row(self, data: LandmarksData) -> list[str]:
        # make the landmarks a list of strings
        return [str(data.lm_valid)] + [f"({i[0]}, {i[1]}, {i[2]})" for i in data.landmarks]

    def draw(self, image: np.ndarray, data: LandmarksData, x_offset: int = 0, y_offset: int = 0) -> np.ndarray:
        """Draw the landmarks on the image"""
        color = (0, 0 , 0)
        for (x, y, *_) in data.landmarks:
            cv2.circle(image, (x+x_offset, y+y_offset), 2, color, -1)
        return image
    
    def compute(self, features: np.ndarray, valid: bool) -> LandmarksData:
        """Compute the 68 landmarks for the image"""
        return LandmarksData(landmarks=features[self.lms_478_to_68], lm_valid=valid)