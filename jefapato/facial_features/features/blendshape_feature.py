__all__ = [
    "Blendshape", 
    "BS_Neutral", 
    "BS_BrowDownLeft", 
    "BS_BrowDownRight", 
    "BS_BrowInnerUp", 
    "BS_BrowOuterUpLeft", 
    "BS_BrowOuterUpRight",
    "BS_CheekPuff",
    "BS_CheekSquintLeft",
    "BS_CheekSquintRight",
    "BS_EyeBlinkLeft",
    "BS_EyeBlinkRight",
    "BS_EyeLookDownLeft",
    "BS_EyeLookDownRight",
    "BS_EyeLookInLeft",
    "BS_EyeLookInRight",
    "BS_EyeLookOutLeft",
    "BS_EyeLookOutRight",
]

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

    def compute(self, features: dict[str, Any], valid: bool) -> BlendshapeData:
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
        
        if not valid:
            return BlendshapeData(blendshape_value=0, blendshape_valid=False, side=self.side)

        # TODO check how to compute the valid state...
        return BlendshapeData(
            blendshape_value=features[self.mediapipe_key],
            blendshape_valid=True,
            side=self.side,
        )

KEYS = [
    'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight',
    'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 
    'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 
    'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 
    'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 
    'noseSneerRight'
]

class BS_Neutral(Blendshape):
    """A class to represent the neutral expression"""
    mediapipe_key = "_neutral"
    side = "whole"

class BS_BrowDownLeft(Blendshape):
    """A class to represent the brow down left expression"""
    mediapipe_key = "browDownLeft"
    side = "left"

class BS_BrowDownRight(Blendshape):
    mediapipe_key = "browDownRight"
    side = "right"

class BS_BrowInnerUp(Blendshape):
    mediapipe_key = "browInnerUp"
    side = "whole"


class BS_BrowOuterUpLeft(Blendshape):
    mediapipe_key = "browOuterUpLeft"
    side = "left"

class BS_BrowOuterUpRight(Blendshape):
    mediapipe_key = "browOuterUpRight"
    side = "right"

class BS_CheekPuff(Blendshape):
    mediapipe_key = "cheekPuff"
    side = "whole"

class BS_CheekSquintLeft(Blendshape):
    mediapipe_key = "cheekSquintLeft"
    side = "left"

class BS_CheekSquintRight(Blendshape):  
    mediapipe_key = "cheekSquintRight"
    side = "right"

class BS_EyeBlinkLeft(Blendshape):
    mediapipe_key = "eyeBlinkLeft"
    side = "left"

class BS_EyeBlinkRight(Blendshape):
    mediapipe_key = "eyeBlinkRight"
    side = "right"

class BS_EyeLookDownLeft(Blendshape):
    mediapipe_key = "eyeLookDownLeft"
    side = "left"

class BS_EyeLookDownRight(Blendshape):
    mediapipe_key = "eyeLookDownRight"
    side = "right"

class BS_EyeLookInLeft(Blendshape):
    mediapipe_key = "eyeLookInLeft"
    side = "left"

class BS_EyeLookInRight(Blendshape):
    mediapipe_key = "eyeLookInRight"
    side = "right"

class BS_EyeLookOutLeft(Blendshape):
    mediapipe_key = "eyeLookOutLeft"
    side = "left"

class BS_EyeLookOutRight(Blendshape):
    mediapipe_key = "eyeLookOutRight"
    side = "right"