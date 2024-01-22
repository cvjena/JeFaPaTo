__all__ = ["InputQueueItem", "AnalyzeQueueItem"]

from dataclasses import dataclass

import numpy as np

@dataclass
class InputQueueItem:
    """A dataclass for holding an input image and its metadata.
    
    It stores the image as a numpy array and the timestamp as a float.
    This is used for the input queue.
    """
    frame: np.ndarray
    timestamp: float
    
@dataclass
class AnalyzeQueueItem:
    """A dataclass for holding an input image and its metadata."""

    image: np.ndarray
    valid: bool
    landmark_features: np.ndarray
    blendshape_features: dict[str, float]
    x_offset: int
    y_offset: int