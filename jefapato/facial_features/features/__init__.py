__all__ = ["Feature", "FeatureData", "EAR2D6", "EAR_Data", "EAR3D6", "Blendshape", "BlendshapeData", "BLENDSHAPES"]


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
]
BLENDSHAPES.sort(key=lambda x: x.mediapipe_key)