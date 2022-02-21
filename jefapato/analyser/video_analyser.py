__all__ = ["VideoAnalyser"]

from typing import Tuple, Type

import cv2
import numpy as np

from jefapato import extracting, loading

from .abstract_analyser import Analyser


class VideoAnalyser(Analyser):
    def __init__(self, extractor_c: Type[extracting.Extractor]) -> None:
        super().__init__(
            loader_c=loading.VideoDataLoader,
            extractor_c=extractor_c,
        )

    def load_resource(self):
        self.resource: cv2.VideoCapture = cv2.VideoCapture(
            self.resource_path.as_posix()
        )
        self.set_data_amount()

    def release_resource(self):
        self.resource.release()

    def get_next_item(self) -> Tuple[bool, np.ndarray]:
        return self.resource.read()

    def set_data_amount(self):
        self.data_amount = self.resource.get(cv2.CAP_PROP_FRAME_COUNT)

    def reset_data_resource(self):
        self.set_next_item_by_id(0)

    def get_fps(self) -> float:
        return self.resource.get(cv2.CAP_PROP_FPS)

    def set_next_item_by_id(self, value: int):
        self.resource.set(cv2.CAP_PROP_POS_FRAMES, value)

    def get_item_size_in_bytes(self):
        width = self.resource.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.resource.get(cv2.CAP_PROP_FRAME_HEIGHT)
        channels = 3
        data_size_in_bytes = 1
        return width * height * channels * data_size_in_bytes

    def get_item_size(self):
        width = self.resource.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.resource.get(cv2.CAP_PROP_FRAME_HEIGHT)
        channels = 3
        return width, height, channels
