__all__ = ["Blendshape", "BS_Neutral", "BS_BrowDownLeft", "BS_BrowDownRight"]

from dataclasses import dataclass
from typing import Any
from .abstract_feature import Feature, FeatureData

@dataclass
class BlendshapeData(FeatureData):
    """A simple data class to combine the ear score results"""

    blendshape_value: float
    blendshape_valid: bool
    side: str


class Blendshape(Feature):
    plot_info = {
        "blendshape_value": {"color": (0, 0, 0)},
    }
    mediapipe_key = "NOT_SET"
    side = "NOT_SET"
    is_blendshape = True

    def get_header(self) -> list[str]:
        """
        Returns
        -------
        List[str]
            The header for the feature including the valid and all the landmarks
        """
        return [f"{self.__class__.__name__}_value", f"{self.__class__.__name__}_valid"]

    def as_row(self, data: BlendshapeData) -> list[str]:
        """Return the data as a row as in the same order as the header"""
        return [f"{data.blendshape_value: .8f}", str(data.blendshape_valid)]

    def compute(self, features: dict[str, Any]) -> BlendshapeData:
        """
        Extract the blendshape value

        Parameters
        ----------
        features : dict[str, FeatureData]
            The features to compute the blendshape value from

        Returns
        -------
        BlendshapeData
            The computed blendshape value
        """
        if self.mediapipe_key == "NOT_SET":
            return BlendshapeData(blendshape_value=0, blendshape_valid=False, side=self.side)

        # TODO check how to compute the valid state...
        return BlendshapeData(
            blendshape_value=features[self.mediapipe_key],
            blendshape_valid=True,
            side=self.side,
        )

KEYS = [
    'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 
    'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 
    'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 
    'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 
    'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 
    'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 
    'noseSneerRight'
]


class BS_Neutral(Blendshape):
    """A class to represent the neutral expression"""

    def __init__(self):
        self.mediapipe_key = "_neutral"
        self.side = "whole"

class BS_BrowDownLeft(Blendshape):
    """A class to represent the brow down left expression"""

    def __init__(self):
        self.mediapipe_key = "browDownLeft"
        self.side = "left"

class BS_BrowDownRight(Blendshape):
    """"""
    def __init__(self):
        self.mediapipe_key = "browDownRight"
        self.side = "right"