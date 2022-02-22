__all__ = ["Feature"]

import abc
from typing import Any, Dict


class Feature(abc.ABC):
    plot_info: Dict[str, Dict[str, Any]] = {}

    def __init__(self) -> None:
        super().__init__()
        self.d_type = None

    @abc.abstractmethod
    def compute(self, in_data: Any) -> Any:
        pass
