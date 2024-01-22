from __future__ import annotations

__all__ = ["Feature", "FeatureData"]

import abc
from typing import Any
import numpy as np

class FeatureData(abc.ABC):
    pass

class Feature(abc.ABC):
    plot_info: dict[str, dict[str, Any]] = {}
    is_blendshape = False # TODO this can be done nicer with hasattr(self, "blendshape_index") check...

    @abc.abstractmethod
    def compute(self, features: Any, valid: bool) -> FeatureData:
        """
        Needs to be implemented by the inheriting class
        """

    @abc.abstractmethod
    def get_header(self) -> list[str]:
        """
        Needs to be implemented by the inheriting class
        """

    @abc.abstractmethod
    def as_row(self, data: FeatureData) -> list[str]:
        """
        Needs to be implemented by the inheriting class
        """

    def draw(self, image: np.ndarray, data: FeatureData, x_offset: int=0, y_offset: int=0) -> None:
        """
        Draw the feature on the given image in place. Note that this is not
        implemented for all features and will do nothing for those.
        Args:
            image (np.ndarray): The image on which to draw the feature.
            data (FeatureData): The data associated with the feature.
            x_offset (int, optional): The x-axis offset for the feature's position. Defaults to 0.
            y_offset (int, optional): The y-axis offset for the feature's position. Defaults to 0.

        Returns:
            None
        """
        return
