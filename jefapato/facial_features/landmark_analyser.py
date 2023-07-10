__all__ = ["FaceAnalyzer"]

import collections
import pathlib
from typing import Any, List, OrderedDict, Tuple, Type, Union, Optional

import cv2
import numpy as np
import pluggy
import psutil
import structlog

from .features import Feature, FeatureData
from .video_data_loader import VideoDataLoader
from .mediapipe_landmark_extractor import MediapipeLandmarkExtractor

logger = structlog.get_logger()

class FaceAnalyzer():
    hookimpl = pluggy.HookimplMarker("analyser")
    hookspec = pluggy.HookspecMarker("analyser")
    def __init__(
        self, 
        max_ram_size: int = 4<<28
    ):
        super().__init__()
        self.max_ram_size = max_ram_size

        self.feature_methods: OrderedDict[str, Feature] = collections.OrderedDict()
        self.feature_data: OrderedDict[str, list[FeatureData]] = collections.OrderedDict()

        self.resource: Any = None
        self.resource_path: Optional[Union[pathlib.Path, int]] = None
        self.data_amount: int = 0

        self.pm = pluggy.PluginManager("analyser")
        self.pm.add_hookspecs(self.__class__)# TODO could be an issue!

    def analysis_setup(self) -> bool:
        if self.resource is None:
            if self.resource_path is None:
                logger.error("Resource could not be created!")
                return False
            self.load_resource()

        # compute how much RAM space we have
        available_memory = min(psutil.virtual_memory().available, self.max_ram_size)
        item_size = self.get_item_size_in_bytes()
        items_to_place = int(available_memory // item_size)

        w, h, d = self.get_item_size()
        b = self.get_item_size_in_bytes()
        logger.info("Item information", width=w, height=h, depth=d, bytes=b)
        logger.info("Available memory", memory=available_memory)
        logger.info("Data loader queue space", space=items_to_place)

        self.loader = VideoDataLoader(self.get_next_item, items_to_place)
        self.extractor = MediapipeLandmarkExtractor(data_queue=self.loader.data_queue, data_amount=self.data_amount)

        self.extractor.processedPercentage.connect(lambda x: self.pm.hook.processed_percentage(percentage=x))
        self.extractor.processingFinished.connect(lambda: self.pm.hook.finished())
        self.extractor.processingFinished.connect(self.release_resource)
        return True

    def analysis_start(self):
        self.pm.hook.started()
        self.loader.start()
        logger.info("Started loader thread.", loader=self.loader)
        self.extractor.start()
        logger.info("Started extractor thread.", extractor=self.extractor)

    def set_resource_path(self, value: Union[int, pathlib.Path]) -> None:
        self.resource_path = value
        self.load_resource()

    def get_resource_path(self) -> Union[pathlib.Path, int, None]:
        return self.resource_path

    def stop(self):
        # TODO create a stop function for the loader and extractor
        #      and don't use the variables directly...
        if self.loader is not None:
            self.loader.stopped = True
            self.loader.join()
        if self.extractor is not None:
            self.extractor.stopped = True
            # self.extractor.terminate()
            self.extractor.wait()

    def reset(self):
        self.features = list()

    def register_hooks(self, plugin: object) -> None:
        self.pm.register(plugin)

    def set_features(self, features: List[Type[Feature]]) -> None:
        self.feature_methods.clear()
        self.feature_data.clear()

        for feature in features:
            self.feature_methods[feature.__name__] = feature(**self.kwargs)
            self.feature_data[feature.__name__] = []

    def set_settings(self, **kwargs) -> None:
        kwargs["backend"] = kwargs.get("backend", "mediapipe")
        self.kwargs = kwargs

    def set_skip_count(self, value: int) -> None:
        self.extractor.set_skip_count(value)

    def toggle_pause(self) -> None:
        self.extractor.toggle_pause()

    def start(self) -> None:
        for m_name in self.feature_methods:
            self.feature_data[m_name].clear()

        self.analysis_setup()
        self.extractor.processingUpdated.connect(self.handle_update)
        self.extractor.processingPaused.connect(self.pm.hook.paused)
        self.extractor.processingResumed.connect(self.pm.hook.resumed)
        self.analysis_start()

    def handle_update(
        self, 
        image: np.ndarray, 
        face_rect: tuple[int, int, int, int], 
        features: np.ndarray
    ) -> None:
        # here would be some drawing? and storing of the features we are interested in
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