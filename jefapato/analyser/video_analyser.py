__all__ = ["VideoAnalyser"]

import pathlib
import platform
from typing import Optional, Tuple, Type

import cv2
import numpy as np

from jefapato import extracting, loading

from .abstract_analyser import Analyser


class VideoAnalyser(Analyser):
    """
    An analyser class for video video data in the form of video files
    and webcam streams.

    Depending on the set resource path, the analyser will load the video
    data from the file or from the webcam.
    """

    def __init__(self, extractor_c: Optional[Type[extracting.Extractor]] = None) -> None:
        super().__init__(
            loader_c=loading.VideoDataLoader,
            extractor_c=extractor_c,
        )

    def load_resource(self):
        resource_type = self.resource_path
        if isinstance(resource_type, pathlib.Path):
            self.resource = cv2.VideoCapture(resource_type.as_posix())
            self.data_amount = self.resource.get(cv2.CAP_PROP_FRAME_COUNT)
        # this is the case for a webcam
        elif isinstance(resource_type, int):
            if platform.system() == "Darwin":
                # check for the architecture
                if platform.processor() == "arm":
                    self.resource = cv2.VideoCapture(1)
                else:
                    self.resource = cv2.VideoCapture(0)
                self.resource.set(cv2.CAP_PROP_FPS, 30)
            else:
                self.resource = cv2.VideoCapture(0)
            # TODO make this configurable
            # with v4l2-ctl --list-formats-ext one chan check if the format is supported
            self.resource.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.resource.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # 1280
            self.resource.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # 720
            self.data_amount = np.inf
        else:
            raise ValueError("The resource path must be a pathlib.Path or an int")

    def release_resource(self):
        self.resource.release()
        self.resource = None

    def get_next_item(self) -> Tuple[bool, np.ndarray]:
        return self.resource.read()

    def get_fps(self) -> float:
        return self.resource.get(cv2.CAP_PROP_FPS)

    def get_item_size_in_bytes(self) -> int:
        width = self.resource.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.resource.get(cv2.CAP_PROP_FRAME_HEIGHT)
        channels = 3
        data_size_in_bytes = 1
        return width * height * channels * data_size_in_bytes

    def get_item_size(self) -> Tuple[int, int, int]:
        width = self.resource.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.resource.get(cv2.CAP_PROP_FRAME_HEIGHT)
        channels = 3
        return width, height, channels
