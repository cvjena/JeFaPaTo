__all__ = ["Landmarks478"]

import dataclasses

import cv2
import numpy as np

from .abstract_feature import Feature, FeatureData


@dataclasses.dataclass
class Landmarks478_Data(FeatureData):
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

    def as_row(self, data: Landmarks478_Data) -> list[str]:
        """Return the data as a row as in the same order as the header"""

        # make the landmarks a list of strings
        return [str(data.lm_valid)] + [f"({i[0]}, {i[1]}, {i[2]})" for i in data.landmarks]

    def draw(self, image: np.ndarray, data: Landmarks478_Data, x_offset: int = 0, y_offset: int = 0) -> np.ndarray:
        """Draw the landmarks on the image"""
        color = (0, 0 , 0)
        for (x, y, *_) in data.landmarks:
            cv2.circle(image, (x+x_offset, y+y_offset), 1, color, -1)
        return image


    def compute(self, features: np.ndarray, valid: bool) -> Landmarks478_Data:
        """Compute the 478 landmarks for the image"""
        return Landmarks478_Data(landmarks=features, lm_valid=valid)