from __future__ import annotations

__all__ = ["Feature", "FeatureData"]

import abc
from typing import Any
import numpy as np

class FeatureData(abc.ABC):
    pass


class Feature(abc.ABC):
    plot_info: dict[str, dict[str, Any]] = {}
    is_blendshape = False

    @abc.abstractmethod
    def compute(self, features: Any, valid: bool) -> FeatureData:
        pass

    @abc.abstractmethod
    def get_header(self) -> list[str]:
        pass

    @abc.abstractmethod
    def as_row(self, data: FeatureData) -> list[str]:
        pass

    def draw(self, image: np.ndarray, data: FeatureData, x_offset: int= 0, y_offset: int=0) -> None:
        pass
