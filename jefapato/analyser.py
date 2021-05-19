from abc import ABC, abstractmethod
from typing import Type, List, Callable, Any, Union, Tuple
from pathlib import Path

from .data_loader import DataLoader, VideoDataLoader
from .feature_extractor import FeatureExtractor
from .classifier import Classifier
from .data_processor import DataProcessor


import cv2
import numpy as np

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

        self.resource: Any = None
        self.resource_path: Union[Path, None] = None
        self.data_amount: int = 0

    def analyse(self, data: Any):
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

    def analysis_start(self):
        self.features = list()
        self.classes = list()
        # reset the data resource to the initial state, like frame 0
        self.reset_data_resource()

        self.data_loader = self.data_loader_class(self.get_next_item)
        self.data_loader.start()

        self.data_processor = DataProcessor(self.analyse, self.get_data_amount(), self.data_loader)
        for f in self.functions_processing_started:
            self.data_processor.processingStarted.connect(f)

        for f in self.functions_processing_updated:
            self.data_processor.processingUpdated.connect(f)
        
        for f in self.functions_processing_finished:
            self.data_processor.processingFinished.connect(f)

        self.data_processor.start()

    def current_feature(self):
        return self.features[-1]

    def current_class(self):
        return self.classes[-1]

    def set_resource_path(self, value: Union[str, Path]):
        if isinstance(value, str):
            value = Path(value)
        self.resource_path = value
        self.load_resource()

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