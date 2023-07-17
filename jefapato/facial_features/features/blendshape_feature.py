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
    "BS_EyeSquintLeft",
    "BS_EyeSquintRight",
    "BS_EyeWideLeft",
    "BS_EyeWideRight",
    "BS_JawForward",
    "BS_JawLeft",
    "BS_JawOpen",
    "BS_JawRight",
    "BS_MouthClose",
    "BS_MouthDimpleLeft",
    "BS_MouthDimpleRight",
    "BS_MouthFrownLeft",
    "BS_MouthFrownRight",
    "BS_MouthFunnel",
    "BS_MouthLeft",
    "BS_MouthLowerDownLeft",
    "BS_MouthLowerDownRight",
    "BS_MouthPressLeft",
    "BS_MouthPressRight",
    "BS_MouthPucker",
    "BS_MouthRight",
    "BS_MouthRollLower",
    "BS_MouthRollUpper",
    "BS_MouthShrugLower",
    "BS_MouthShrugUpper",
    "BS_MouthSmileLeft",
    "BS_MouthSmileRight",
    "BS_MouthStretchLeft",
    "BS_MouthStretchRight",
    "BS_MouthUpperUpLeft",
    "BS_MouthUpperUpRight",
    "BS_NoseSneerLeft",
    "BS_NoseSneerRight"
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

class BS_EyeLookUpLeft(Blendshape):
    mediapipe_key = "eyeLookUpLeft"
    side = "left"

class BS_EyeLookUpRight(Blendshape):
    mediapipe_key = "eyeLookUpRight"
    side = "right"

class BS_EyeSquintLeft(Blendshape): 
    mediapipe_key = "eyeSquintLeft"
    side = "left"

class BS_EyeSquintRight(Blendshape):
    mediapipe_key = "eyeSquintRight"
    side = "right"

class BS_EyeWideLeft(Blendshape):
    mediapipe_key = "eyeWideLeft"
    side = "left"

class BS_EyeWideRight(Blendshape):
    mediapipe_key = "eyeWideRight"
    side = "right"

class BS_JawForward(Blendshape):
    mediapipe_key = "jawForward"
    side = "whole"

class BS_JawLeft(Blendshape):
    mediapipe_key = "jawLeft"
    side = "whole"

class BS_JawOpen(Blendshape):
    mediapipe_key = "jawOpen"
    side = "whole"

class BS_JawRight(Blendshape):
    mediapipe_key = "jawRight"
    side = "whole"

class BS_MouthClose(Blendshape):
    mediapipe_key = "mouthClose"
    side = "whole"

class BS_MouthDimpleLeft(Blendshape):
    mediapipe_key = "mouthDimpleLeft"
    side = "left"

class BS_MouthDimpleRight(Blendshape):
    mediapipe_key = "mouthDimpleRight"
    side = "right"

class BS_MouthFrownLeft(Blendshape):
    mediapipe_key = "mouthFrownLeft"
    side = "left"

class BS_MouthFrownRight(Blendshape):
    mediapipe_key = "mouthFrownRight"
    side = "right"

class BS_MouthFunnel(Blendshape):
    mediapipe_key = "mouthFunnel"
    side = "whole"

class BS_MouthLeft(Blendshape):
    mediapipe_key = "mouthLeft"
    side = "left"

class BS_MouthLowerDownLeft(Blendshape):
    mediapipe_key = "mouthLowerDownLeft"
    side = "left"

class BS_MouthLowerDownRight(Blendshape):
    mediapipe_key = "mouthLowerDownRight"
    side = "right"

class BS_MouthPressLeft(Blendshape):
    mediapipe_key = "mouthPressLeft"
    side = "left"

class BS_MouthPressRight(Blendshape):
    mediapipe_key = "mouthPressRight"
    side = "right"

class BS_MouthPucker(Blendshape):
    mediapipe_key = "mouthPucker"
    side = "whole"

class BS_MouthRight(Blendshape):
    mediapipe_key = "mouthRight"
    side = "right"

class BS_MouthRollLower(Blendshape):
    mediapipe_key = "mouthRollLower"
    side = "whole"

class BS_MouthRollUpper(Blendshape):
    mediapipe_key = "mouthRollUpper"
    side = "whole"

class BS_MouthShrugLower(Blendshape):
    mediapipe_key = "mouthShrugLower"
    side = "whole"

class BS_MouthShrugUpper(Blendshape):
    mediapipe_key = "mouthShrugUpper"
    side = "whole"

class BS_MouthSmileLeft(Blendshape):
    mediapipe_key = "mouthSmileLeft"
    side = "left"

class BS_MouthSmileRight(Blendshape):
    mediapipe_key = "mouthSmileRight"
    side = "right"

class BS_MouthStretchLeft(Blendshape):
    mediapipe_key = "mouthStretchLeft"
    side = "left"

class BS_MouthStretchRight(Blendshape):
    mediapipe_key = "mouthStretchRight"
    side = "right"

class BS_MouthUpperUpLeft(Blendshape):
    mediapipe_key = "mouthUpperUpLeft"
    side = "left"

class BS_MouthUpperUpRight(Blendshape):
    mediapipe_key = "mouthUpperUpRight"
    side = "right"

class BS_NoseSneerLeft(Blendshape):
    mediapipe_key = "noseSneerLeft"
    side = "left"

class BS_NoseSneerRight(Blendshape):
    mediapipe_key = "noseSneerRight"
    side = "right"
