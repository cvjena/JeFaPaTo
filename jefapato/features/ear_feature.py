__all__ = ["EARFeature", "EARData"]

import dataclasses

import numpy as np
from scipy.spatial import distance

from .abstract_feature import Feature


@dataclasses.dataclass
class EARData:
    """A simple data class to combine the classifcation results"""

    lm_l: np.ndarray
    lm_r: np.ndarray
    ear_l: float
    ear_r: float
    valid: bool


class EARFeature(Feature):
    """EyeBlinking Classifier Class for 68 facial landmarks features

    This class implements a classifier specificly for the 68 landmarking
    schemata.
    """

    def __init__(self, backend: str = "dlib") -> None:
        super().__init__()

        if backend == "dlib":
            self.index_eye_l: slice = slice(42, 48)
            self.index_eye_r: slice = slice(36, 42)
        else:
            raise NotImplementedError("Currently only dlib is supported")

    def ear_score(self, eye: np.ndarray) -> float:
        """Compute the EAR Score for eye landmarks

        Soukupová and Čech 2016: Real-Time Eye Blink Detection using Facial Landmarks
        http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf

        ear = (|p2 - p6| + |p3 - p5| )/ 2|p1-p4|
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

    def compute(self, in_data: np.ndarray) -> EARData:
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
        lm_l = in_data[self.index_eye_l]
        lm_r = in_data[self.index_eye_r]
        valid = not (
            np.allclose(np.zeros_like(lm_l), lm_l)
            and np.allclose(np.zeros_like(lm_r), lm_r)
        )
        ear_l = self.ear_score(lm_l) if valid else 1.0
        ear_r = self.ear_score(lm_r) if valid else 1.0
        return EARData(lm_l, lm_r, ear_l, ear_r, valid)

    # def check_closing(self, score_left, score_right, threshold) -> Tuple[bool, bool]:
    #     """Check if eyes are closed

    #     Compute if the eyes are closed based on a given threshhold

    #     Args:
    #         score_left ([type]): EAR score for the left eye
    #         score_right ([type]): EAR score for the right eye
    #         threshold ([type]): threshold for eye closing

    #     Returns:
    #         Tuple[bool, bool]: closing state for left and right eye
    #     """
    #     return (score_left < threshold, score_right < threshold)

    # def classify(
    #     self, features: List[Tuple[dlib.rectangle, np.ndarray]]
    # ) -> List[EyeBlinkingResult]:
    #     """Classify on the given features

    #     Args:
    #         features (List[Tuple[dlib.rectangle, np.ndarray]]): features of a have
    # with
    #             the face bounding box and the 68 facial feature landmarks

    #     Returns:
    #         List[EyeBlinkingResult]: List of EyeBlinking clasifiction (list entry for
    #             each face), will be invalid EyeBlinking object if features where not
    #             valid
    #     """

    #     # if features is None, it means the feature extracture could not
    #     # create the featureas we wanted, hence, we return an value which is not
    #     # possible!
    #     if not features:
    #         return [EyeBlinkingResult(1.0, 1.0, False)]

    #     classes: List[EyeBlinkingResult] = list()

    #     for _, shape in features:
    #         ear_left = self.ear_score(shape[self.eye_left_slice])
    #         ear_right = self.ear_score(shape[self.eye_right_slice])
    #         classes.append(EyeBlinkingResult(ear_left, ear_right, True))

    #     return classes
