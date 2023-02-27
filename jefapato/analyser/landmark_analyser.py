__all__ = ["LandmarkAnalyser"]

import collections
from typing import Any, List, OrderedDict, Type

import cv2

# import dlib
import numpy as np
import structlog

from jefapato import extracting, features

from .abstract_analyser import hookspec
from .video_analyser import VideoAnalyser

logger = structlog.get_logger()


class LandmarkAnalyser(VideoAnalyser):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.kwargs = kwargs

        self.feature_methods: OrderedDict[
            str, features.Feature
        ] = collections.OrderedDict()
        self.feature_data: OrderedDict[
            str, features.FeatureData
        ] = collections.OrderedDict()

    def set_features(self, features: List[Type[features.Feature]]) -> None:
        self.feature_methods.clear()
        self.feature_data.clear()

        for feature in features:
            self.feature_methods[feature.__name__] = feature(**self.kwargs)
            self.feature_data[feature.__name__] = []

    def set_settings(self, **kwargs) -> None:
        kwargs["backend"] = kwargs.get("backend", "dlib")
        self.kwargs = kwargs

    def set_skip_count(self, value: int) -> None:
        self.extractor.set_skip_count(value)

    def toggle_pause(self) -> None:
        self.extractor.toggle_pause()

    def start(self) -> None:
        # this should be done in a nicer way...
        if self.kwargs["backend"] == "dlib":
            self.extractor_c = extracting.DlibLandmarkExtractor
        elif self.kwargs["backend"] == "mediapipe":
            self.extractor_c = extracting.MediapipeLandmarkExtractor
        else:
            logger.error("Only dlib backend is supported in LandmarkAnalyser")
            return

        for m_name in self.feature_methods:
            self.feature_data[m_name].clear()

        self.analysis_setup()
        self.extractor.processingUpdated.connect(self.handle_update)
        self.extractor.processingPaused.connect(self.pm.hook.paused)
        self.extractor.processingResumed.connect(self.pm.hook.resumed)
        self.analysis_start()

    def handle_update(
        self, image: np.ndarray, face_rect: Any, features: np.ndarray
    ) -> None:
        # here would be some drawing? and storing of the features we are interested in
        face_rect: tuple[int, int, int, int] = face_rect
        temp_data = collections.OrderedDict()

        for f_name, f_class in self.feature_methods.items():
            res = f_class.compute(features)
            image = res.draw(image)
            self.feature_data[f_name].append(res)
            temp_data[f_name] = res

        self.pm.hook.updated_feature(feature_data=temp_data)

        face = image[face_rect[1] : face_rect[3], face_rect[0] : face_rect[2]]
        cv2.rectangle(
            image,
            (face_rect[0], face_rect[1]),
            (face_rect[2], face_rect[3]),
            (0, 255, 0),
            2,
        )
        self.pm.hook.updated_display(image=image, face=face)

    @hookspec
    def updated_display(self, image: np.ndarray, face: np.ndarray):
        """
        Trigger a hook that the displaying information was updated.
        """

    @hookspec
    def updated_feature(self, feature_data: OrderedDict[str, Any]) -> None:
        """
        Trigger a hook that the features were updated.
        """

    def get_header(self) -> List[str]:
        header = ["frame"]
        for v in self.feature_methods.values():
            header.extend(v.get_header())

        return header

    def __iter__(self):
        self.__iter_counter = 0
        return self

    def __next__(self):
        keys = [*self.feature_data]  # unpack the keys to a list :^)
        if len(keys) == 0:
            raise StopIteration

        if self.__iter_counter >= len(self.feature_data[keys[0]]):
            raise StopIteration

        row = [self.__iter_counter]
        self.__iter_counter += 1
        for v in self.feature_data.values():
            row.extend(v[self.__iter_counter - 1].as_row())

        return row

    def call_after_resource_load(self) -> None:
        pass
