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
    ear_left: float
    ear_right: float
    closed_left: bool
    closed_right: bool


class EyeBlinking68Classifier(Classifier):
    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold: float = threshold

        self.eye_left_slice: slice = slice(42, 48)
        self.eye_right_slice: slice = slice(36, 42)

    def ear_score(self, eye: np.ndarray): 
        # Soukupová and Čech 2016: Real-Time Eye Blink Detection using Facial Landmarks
        # http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
        #
        # ear = (|p2 - p6| + |p3 - p5| )/ 2|p1-p4|
        #            A           B           C

        # dont forget the 0-index
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])

        return (A + B) / (2.0 * C)

    def check_closing(self, score_left, score_right, threshold):
        return (score_left < threshold, score_right < threshold)

    def classify(self, features: List[Tuple[dlib.rectangle, np.ndarray]]) -> Any:

        classes: List[EyeBlinkingResult] = list()

        for _ , shape in features:
            eye_left  = shape[self.eye_left_slice]
            eye_right = shape[self.eye_right_slice]

            ear_left  = self.ear_score(eye_left)
            ear_right = self.ear_score(eye_right)

            closed_left, closed_right = self.check_closing(ear_left, ear_right, self.threshold)

            classes.append(EyeBlinkingResult(ear_left, ear_right, closed_left, closed_right))

        return classes