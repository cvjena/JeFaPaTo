__all__ = ["Feature"]

import abc
from typing import Any


class Feature(abc.ABC):
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def compute(self, in_data: Any) -> Any:
        pass
