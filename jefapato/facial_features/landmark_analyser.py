__all__ = ["FaceAnalyzer"]

from typing import Any, Type
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import pluggy
import psutil
import structlog
import platform

from .features import Feature, FeatureData
from .video_data_loader import VideoDataLoader
from .mediapipe_landmark_extractor import MediapipeLandmarkExtractor
from .queue_items import AnalyzeQueueItem

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

        self.feature_methods: OrderedDict[str, Feature] = OrderedDict()
        self.feature_data: OrderedDict[str, list[FeatureData]] = OrderedDict()

        self.resource_interface: Any = None
        self.video_resource: Path | int | None = None
        self.data_amount: int = 0

        self.pm = pluggy.PluginManager("analyser")
        self.pm.add_hookspecs(self.__class__)# TODO could be an issue!

    def analysis_setup(self) -> bool:
        if self.resource_interface is None:
            logger.error("No resource interface set.")
            raise ValueError("No resource interface set.")

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


    def stop(self):
        if (loader := getattr(self, "loader", None)) is not None:
            loader.stopped = True
            loader.join()
        if (extractor := getattr(self, "extractor", None)) is not None:
            extractor.stopped = True
            extractor.wait()

    def reset(self):
        self.features = list()

    def register_hooks(self, plugin: object) -> None:
        self.pm.register(plugin)

    def set_features(self, features: list[Type[Feature]]) -> None:
        self.feature_methods.clear()
        self.feature_data.clear()

        for feature in features:
            self.feature_methods[feature.__name__] = feature()
            self.feature_data[feature.__name__] = []

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

    def handle_update(self, q_item: AnalyzeQueueItem) -> None:
        # here would be some drawing? and storing of the features we are interested in
        temp_data = OrderedDict()

        image = q_item.image
        face_rect = q_item.face_rect
        features = q_item.landmark_features

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

    def get_header(self) -> list[str]:
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

        row = [str(self.__iter_counter)]
        self.__iter_counter += 1
        for v in self.feature_data.values():
            row.extend([x for x in v[self.__iter_counter - 1].as_row()])

        return row

    def prepare_video_resource(self, value: Path | int):
        self.video_resource = value

        if not isinstance(self.video_resource, (Path, int)):
            raise ValueError("Video resource must be a Path or an integer.")

        if isinstance(self.video_resource, Path):
            if not self.video_resource.exists():
                raise FileNotFoundError(f"File {self.video_resource} does not exist.")

            self.resource_interface = cv2.VideoCapture(str(self.video_resource.absolute()))
            self.data_amount = self.resource_interface.get(cv2.CAP_PROP_FRAME_COUNT)
            return

        if self.video_resource != -1:
            raise ValueError("Video resource must be a Path or -1 for webcam.")
        
        # this is the case for a webcam
        # check on which kind of system we are running 
        self.resource_interface = cv2.VideoCapture(1 if platform.system() == "Darwin" else 0)
        self.resource_interface.set(cv2.CAP_PROP_FPS, 30)
        self.resource_interface.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.resource_interface.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # 1280 # TODO check if this is correct
        self.resource_interface.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # 720 # TODO check if this is correct
        self.data_amount = -1

    def release_resource(self):
        self.resource_interface.release()
        self.resource_interface = None

    def get_next_item(self) -> tuple[bool, np.ndarray]:
        return self.resource_interface.read()

    def get_fps(self) -> float:
        return self.resource_interface.get(cv2.CAP_PROP_FPS)

    def get_item_size_in_bytes(self) -> int:
        width = self.resource_interface.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.resource_interface.get(cv2.CAP_PROP_FRAME_HEIGHT)
        channels = 3
        data_size_in_bytes = 1
        return width * height * channels * data_size_in_bytes

    def get_item_size(self) -> tuple[int, int, int]:
        width = self.resource_interface.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.resource_interface.get(cv2.CAP_PROP_FRAME_HEIGHT)
        channels = 3
        return width, height, channels

    def get_throughput(self) -> tuple[int, int]:
        data_input = self.loader.processing_per_second
        data_proce = self.extractor.processing_per_second

        return data_input, data_proce