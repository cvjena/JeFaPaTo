from abc import ABC, abstractmethod
from typing import Any, List, Tuple
from dataclasses import dataclass

import dlib
import numpy as np
from scipy.spatial import distance


class Classifier(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def classify(self, features: Any) -> Any:
        pass


@dataclass
class EyeBlinkingResult:
    """A simple data class to combine the classifcation results"""

    ear_left: float
    ear_right: float
    closed_left: bool
    closed_right: bool
    valid: bool


class EyeBlinking68Classifier(Classifier):
    """EyeBlinking Classifier Class for 68 facial landmarks features

    This class implements a classifier specificly for the 68 landmarking
    schemata.
    """

    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold: float = threshold

        self.eye_left_slice: slice = slice(42, 48)
        self.eye_right_slice: slice = slice(36, 42)

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

    def check_closing(self, score_left, score_right, threshold) -> Tuple[bool, bool]:
        """Check if eyes are closed

        Compute if the eyes are closed based on a given threshhold

        Args:
            score_left ([type]): EAR score for the left eye
            score_right ([type]): EAR score for the right eye
            threshold ([type]): threshold for eye closing

        Returns:
            Tuple[bool, bool]: closing state for left and right eye
        """
        return (score_left < threshold, score_right < threshold)

    def classify(
        self, features: List[Tuple[dlib.rectangle, np.ndarray]]
    ) -> List[EyeBlinkingResult]:
        """Classify on the given features

        Args:
            features (List[Tuple[dlib.rectangle, np.ndarray]]): features of a have with
                the face bounding box and the 68 facial feature landmarks

        Returns:
            List[EyeBlinkingResult]: List of EyeBlinking clasifiction (list entry for
                each face), will be invalid EyeBlinking object if features where not
                valid
        """

        # if features is None, it means the feature extracture could not
        # create the featureas we wanted, hence, we return an value which is not
        # possible!
        if not features:
            return [EyeBlinkingResult(1.0, 1.0, False, False, False)]

        classes: List[EyeBlinkingResult] = list()

        for _, shape in features:
            eye_left = shape[self.eye_left_slice]
            eye_right = shape[self.eye_right_slice]

            ear_left = self.ear_score(eye_left)
            ear_right = self.ear_score(eye_right)

            closed_left, closed_right = self.check_closing(
                ear_left, ear_right, self.threshold
            )

            classes.append(
                EyeBlinkingResult(ear_left, ear_right, closed_left, closed_right, True)
            )

        return classes
