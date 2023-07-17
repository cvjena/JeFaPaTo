__all__ = ["Feature", "FeatureData", "EAR2D6", "EAR_Data", "EAR3D6", "Blendshape", "BlendshapeData", "BLENDSHAPES", "BS_Valid", "BlendshapeValidData"]


from jefapato.facial_features.features.abstract_feature import Feature, FeatureData
from jefapato.facial_features.features.ear_feature import EAR_Data, EAR2D6, EAR3D6
from jefapato.facial_features.features.blendshape_feature import Blendshape, BlendshapeData

from jefapato.facial_features.features.blendshape_feature import (
    BS_Neutral, 
    BS_BrowDownLeft, BS_BrowDownRight,
    BS_BrowInnerUp,
    BS_BrowOuterUpLeft, BS_BrowOuterUpRight,
    BS_CheekPuff, BS_CheekSquintLeft, BS_CheekSquintRight,
    BS_EyeBlinkLeft, BS_EyeBlinkRight,
    BS_EyeLookDownLeft, BS_EyeLookDownRight,
    BS_EyeLookInLeft, BS_EyeLookInRight,
    BS_EyeLookOutLeft, BS_EyeLookOutRight,
    BS_EyeSquintLeft, BS_EyeSquintRight,
    BS_EyeWideLeft, BS_EyeWideRight,
    BS_JawForward, BS_JawLeft, BS_JawOpen, BS_JawRight,
    BS_MouthClose,
    BS_MouthDimpleLeft, BS_MouthDimpleRight,
    BS_MouthFrownLeft, BS_MouthFrownRight,
    BS_MouthFunnel,
    BS_MouthLeft,
    BS_MouthLowerDownLeft, BS_MouthLowerDownRight,
    BS_MouthPressLeft, BS_MouthPressRight,
    BS_MouthPucker,
    BS_MouthRight,
    BS_MouthRollLower,
    BS_MouthRollUpper,
    BS_MouthShrugLower,
    BS_MouthShrugUpper,
    BS_MouthSmileLeft, BS_MouthSmileRight,
    BS_MouthStretchLeft, BS_MouthStretchRight,
    BS_MouthUpperUpLeft, BS_MouthUpperUpRight,
    BS_NoseSneerLeft, BS_NoseSneerRight,
    BS_Valid,
    BlendshapeValidData,
)


BLENDSHAPES: list[type[Blendshape]] = [
    BS_Neutral, 
    BS_BrowDownLeft, BS_BrowDownRight,
    BS_BrowInnerUp,
    BS_BrowOuterUpLeft, BS_BrowOuterUpRight,
    BS_CheekPuff, BS_CheekSquintLeft, BS_CheekSquintRight,
    BS_EyeBlinkLeft, BS_EyeBlinkRight,
    BS_EyeLookDownLeft, BS_EyeLookDownRight,
    BS_EyeLookInLeft, BS_EyeLookInRight,
    BS_EyeLookOutLeft, BS_EyeLookOutRight,
    BS_EyeSquintLeft, BS_EyeSquintRight,
    BS_EyeWideLeft, BS_EyeWideRight,
    BS_JawForward, BS_JawLeft, BS_JawOpen, BS_JawRight,
    BS_MouthClose,
    BS_MouthDimpleLeft, BS_MouthDimpleRight,
    BS_MouthFrownLeft, BS_MouthFrownRight,
    BS_MouthFunnel,
    BS_MouthLeft,
    BS_MouthLowerDownLeft, BS_MouthLowerDownRight,
    BS_MouthPressLeft, BS_MouthPressRight,
    BS_MouthPucker,
    BS_MouthRight,
    BS_MouthRollLower,
    BS_MouthRollUpper,
    BS_MouthShrugLower,
    BS_MouthShrugUpper,
    BS_MouthSmileLeft, BS_MouthSmileRight,
    BS_MouthStretchLeft, BS_MouthStretchRight,
    BS_MouthUpperUpLeft, BS_MouthUpperUpRight,
    BS_NoseSneerLeft, BS_NoseSneerRight
]

# Sort blendshapes by mediapipe key but ignore Left and Right during sorting
BLENDSHAPES.sort(key=lambda x: x.mediapipe_key.replace("Left", "").replace("Right", ""))