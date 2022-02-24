from __future__ import annotations

__all__ = ["Feature", "FeatureData"]

import abc
from typing import Any, Dict, List

import numpy as np


class Feature(abc.ABC):
    plot_info: Dict[str, Dict[str, Any]] = {}

    def __init__(self) -> None:
        super().__init__()
        self.d_type = None

    @abc.abstractmethod
    def compute(self, in_data: Any) -> FeatureData:
        pass

    @abc.abstractmethod
    def get_header(self) -> List[str]:
        pass


class FeatureData(abc.ABC):
    @abc.abstractmethod
    def as_row(self) -> List[str]:
        pass

    @abc.abstractmethod
    def draw(self, image: np.ndarray) -> np.ndarray:
        pass
