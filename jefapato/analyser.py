from abc import ABC, abstractmethod
from typing import Optional, Type, List, Callable, Any, Union, Tuple
from pathlib import Path

from .data_loader import DataLoader, VideoDataLoader
from .feature_extractor import FeatureExtractor
from .classifier import Classifier
from .data_processor import DataProcessor


import cv2
import numpy as np
import psutil
import logging

MAX_RAM_SIZE = 4 << 30 # ~4GB

class Analyser(ABC):
    def __init__(self, data_loader_class: Type[DataLoader], feature_extractor, classifier) -> None:
        super().__init__()
        self.data_loader_class: Type[DataLoader] = data_loader_class
        self.feature_extractor: FeatureExtractor = feature_extractor
        self.classifier: Classifier = classifier
        
        self.features: List[Any] = list()
        self.classes:  List[Any] = list()

        self.functions_processing_started:  List[Callable] = list()
        self.functions_processing_updated:  List[Callable] = list()
        self.functions_processing_finished: List[Callable] = list()
        self.functions_processed_percentage: List[Callable] = list()

        self.resource: Any = None
        self.resource_path: Union[Path, None] = None
        self.data_amount: int = 0
        self.logger = logging.getLogger("analyser")

    def analyse(self, data: Any) -> None:
        # TODO allow more feature extractors and classifier on the same data?
        features = self.feature_extractor.extract_features(data)
        self.features.append(features)
        classes = self.classifier.classify(features)
        self.classes.append(classes)

    def resource_path_is_set(self):
        return isinstance(self.resource_path, Path)

    @abstractmethod
    def load_resource(self):
        pass

    @abstractmethod
    def release_resource(self):
        pass

    @abstractmethod
    def get_next_item(self):
        pass

    @abstractmethod
    def set_data_amount(self):
        pass

    @abstractmethod
    def reset_data_resource(self):
        pass

    @abstractmethod
    def set_next_item_by_id(self, value: int):
        pass

    @abstractmethod
    def get_item_size_in_bytes(self):
        pass

    @abstractmethod
    def get_item_size(self):
        pass

    def analysis_start(self):
        self.features = list()
        self.classes = list()
        # reset the data resource to the initial state, like frame 0
        self.reset_data_resource()

        # compute how much RAM space we have
        available_memory = min(psutil.virtual_memory().available, MAX_RAM_SIZE)
        item_size = self.get_item_size_in_bytes()
        items_to_place = available_memory // item_size
        
        self.logger.info(f"Item dimesion: {self.get_item_size()}")
        self.logger.info(f"Item size in bytes: {self.get_item_size_in_bytes()}")
        self.logger.info(f"Available memory: {available_memory}")
        self.logger.info(f"Data loader queue size: {items_to_place} items")

        self.data_loader = self.data_loader_class(self.get_next_item, items_to_place)
        self.data_loader.start()
        
        self.logger.info(f"Started data loader thread.")

        self.data_processor = DataProcessor(self.analyse, self.get_data_amount(), self.data_loader)
        for f in self.functions_processing_started:
            self.data_processor.processingStarted.connect(f)

        for f in self.functions_processing_updated:
            self.data_processor.processingUpdated.connect(f)
        
        for f in self.functions_processing_finished:
            self.data_processor.processingFinished.connect(f)

        for f in self.functions_processed_percentage:
            self.data_processor.processedPercentage.connect(f)

        self.data_processor.start()
        self.logger.info(f"Stared data processor thread.")

    def current_feature(self):
        return self.features[-1]

    def current_class(self):
        return self.classes[-1]

    def set_resource_path(self, value: Union[str, Path]) -> Optional[Any]:
        if isinstance(value, str):
            value = Path(value)
        self.resource_path = value
        self.load_resource()
        read, data = self.get_next_item()
        # set to inital index again because we just loaded it
        self.set_next_item_by_id(0)

        return data

    def get_resource_path(self) -> Union[Path, None]:
        return self.resource_path

    def get_data_amount(self) -> int:
        return self.data_amount

    def connect_on_started(self, functions: List[Callable]) -> None:
        self.functions_processing_started += functions

    def connect_on_updated(self, functions: List[Callable]) -> None:
        self.functions_processing_updated += functions

    def connect_on_finished(self, functions: List[Callable]) -> None:
        self.functions_processing_finished += functions

    def connect_processed_percentage(self, functions: List[Callable]) -> None:
        self.functions_processed_percentage += functions

    def resource_is_loaded(self):
        return self.resource is not None

    def stop(self):
        self.data_loader.stopped = True
        self.data_processor.stopped = True

class ResourcePathNotSetException(Exception):
    pass


class VideoAnalyser(Analyser):
    def __init__(self, feature_extractor: FeatureExtractor, classifier: Classifier) -> None:
        super().__init__(
            data_loader_class=VideoDataLoader,
            feature_extractor=feature_extractor,
            classifier=classifier
            )

    def load_resource(self):
        self.resource: cv2.VideoCapture = cv2.VideoCapture(self.resource_path.as_posix())
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
        width    = self.resource.get(cv2.CAP_PROP_FRAME_WIDTH)
        height   = self.resource.get(cv2.CAP_PROP_FRAME_HEIGHT)
        channels = 3
        data_size_in_bytes = 1
        return width * height * channels * data_size_in_bytes

    def get_item_size(self):
        width    = self.resource.get(cv2.CAP_PROP_FRAME_WIDTH)
        height   = self.resource.get(cv2.CAP_PROP_FRAME_HEIGHT)
        channels = 3
        return width, height, channels