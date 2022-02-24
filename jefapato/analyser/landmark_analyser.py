__all__ = ["LandmarkAnalyser"]

import collections
from typing import Any, List, OrderedDict, Type

import dlib
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

        self.feature_methods = collections.OrderedDict()
        self.feature_data = collections.OrderedDict()

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

    def start(self) -> None:
        # this should be done in a nicer way...
        if self.kwargs["backend"] == "dlib":
            self.extractor_c = extracting.LandmarkExtractor
        else:
            logger.error("Only dlib backend is supported in LandmarkAnalyser")
            return

        for m_name in self.feature_methods:
            self.feature_data[m_name].clear()

        self.analysis_setup()
        self.extractor.processingUpdated.connect(self.handle_update)
        self.analysis_start()

    def handle_update(
        self, data: np.ndarray, face_rect: Any, features: np.ndarray
    ) -> None:
        # here would be some drawing? and storing of the features we are interested in
        if self.kwargs["backend"] == "dlib":
            face_rect: dlib.rectangle = face_rect
        else:
            raise NotImplementedError("Only dlib backend is supported")

        face = data[
            face_rect.top() : face_rect.bottom(), face_rect.left() : face_rect.right()
        ]
        self.pm.hook.updated_display(image=data, face=face)

        temp_data = collections.OrderedDict()

        for m_name, m_class in self.feature_methods.items():
            res = m_class.compute(features)
            self.feature_data[m_name].append(res)
            temp_data[m_name] = res

        self.pm.hook.updated_feature(feature_data=temp_data)

    @hookspec
    def updated_display(self, image: np.ndarray, face: np.ndarray):
        """
        Inform however needs it that the anlysis has updated.
        """

    @hookspec
    def updated_feature(self, feature_data: OrderedDict[str, Any]) -> None:
        pass

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
