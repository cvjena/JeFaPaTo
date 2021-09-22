import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Type, Union

import cv2
import dlib
import numpy as np
import psutil

from jefapato.plotter import EyeBlinkingPlotter

from .classifier import Classifier, EyeBlinking68Classifier, EyeBlinkingResult
from .data_loader import DataLoader, VideoDataLoader
from .data_processor import DataProcessor
from .feature_extractor import FeatureExtractor, LandMarkFeatureExtractor, scale_bbox

MAX_RAM_SIZE = 4 << 30  # ~4GB


class Analyser(ABC):
    def __init__(
        self, data_loader_class: Type[DataLoader], feature_extractor, classifier
    ) -> None:
        super().__init__()
        self.data_loader_class: Type[DataLoader] = data_loader_class
        self.feature_extractor: FeatureExtractor = feature_extractor
        self.classifier: Classifier = classifier

        self.features: List[Any] = list()
        self.classes: List[Any] = list()

        self.functions_processing_started: List[Callable] = list()
        self.functions_processing_updated: List[Callable] = list()
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

        self.logger.info("Started data loader thread.")

        self.data_processor = DataProcessor(
            self.analyse, self.get_data_amount(), self.data_loader
        )
        for f in self.functions_processing_started:
            self.data_processor.processingStarted.connect(f)

        for f in self.functions_processing_updated:
            self.data_processor.processingUpdated.connect(f)

        for f in self.functions_processing_finished:
            self.data_processor.processingFinished.connect(f)

        for f in self.functions_processed_percentage:
            self.data_processor.processedPercentage.connect(f)

        self.data_processor.start()
        self.logger.info("Stared data processor thread.")

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
    def __init__(
        self, feature_extractor: FeatureExtractor, classifier: Classifier
    ) -> None:
        super().__init__(
            data_loader_class=VideoDataLoader,
            feature_extractor=feature_extractor,
            classifier=classifier,
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


class EyeBlinkingVideoAnalyser(VideoAnalyser):
    def __init__(self, plotter: EyeBlinkingPlotter) -> None:

        self.eye_blinking_classifier = EyeBlinking68Classifier(threshold=0.2)
        super().__init__(
            feature_extractor=LandMarkFeatureExtractor(),
            classifier=self.eye_blinking_classifier,
        )

        self.results_file_header = (
            "closed_left;closed_right;ear_score_left;ear_score_rigth;valid\n"
        )

        self.plotter: EyeBlinkingPlotter = plotter

        self.connect_on_started([self.__on_start])
        self.connect_on_updated([self.__on_update])
        self.connect_on_finished([self.__on_finished])

        self.score_eye_left: List[float] = list()
        self.score_eye_right: List[float] = list()

        self.closed_eye_left: List[bool] = list()
        self.closed_eye_right: List[bool] = list()
        self.valid: List[bool] = list()

        self.current_frame: int = 0

        if self.plotter is not None:
            self.curve_l = self.plotter.widget_graph.add_curve(
                {"color": "#00F", "width": 2}
            )
            self.curve_r = self.plotter.widget_graph.add_curve(
                {"color": "#F00", "width": 2}
            )

            self.plotter.widget_graph.signal_graph_clicked.connect(
                self.set_current_frame
            )
            self.plotter.widget_graph.signal_x_ruler_changed.connect(
                self.set_current_frame
            )
            self.plotter.widget_graph.signal_y_ruler_changed.connect(self.set_threshold)

    def set_threshold(self, value: float):
        self.eye_blinking_classifier.threshold = value
        self.plotter.widget_graph.set_y_ruler(value)

    def get_threshold(self) -> float:
        return self.eye_blinking_classifier.threshold

    def __on_start(self):
        self.score_eye_left = list()
        self.score_eye_right = list()
        self.closed_eye_left = list()
        self.closed_eye_right = list()
        self.valid = list()
        self.plotter.widget_graph.start(self.get_fps())

    def __on_finished(self):
        self.plotter.widget_graph.finish(self.current_frame)
        self.save_results()

    def __on_update(self, frame: np.ndarray, frame_id: int):
        # currently we only check the first one
        # TODO make possible for more faces
        self.current_frame = frame_id

        self.append_classification()
        self.update_frame(frame)
        self.update_detail(frame)
        self.update_graph()

    def append_classification(self) -> None:
        latest_value: EyeBlinkingResult = self.classes[-1][0]
        self.score_eye_left.append(latest_value.ear_left)
        self.score_eye_right.append(latest_value.ear_right)
        self.closed_eye_left.append(latest_value.closed_left)
        self.closed_eye_right.append(latest_value.closed_right)
        self.valid.append(latest_value.valid)

    def update_graph(self) -> None:
        self.plotter.widget_graph.update(self.current_frame)

        if self.curve_l is not None:
            self.curve_l.setData(self.score_eye_left)
        if self.curve_r is not None:
            self.curve_r.setData(self.score_eye_right)

    def update_frame(self, frame: np.ndarray) -> None:
        img = np.copy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.plotter.widget_frame.frame.set_image(img)

    def set_current_frame(self, frame: float) -> None:
        # clip the frame into the maximum valid range
        # cast to int, just to be sure
        value = int(
            max(
                0,
                min(int(frame), self.data_amount - 1),
            )
        )
        self.plotter.widget_graph.set_x_ruler(value)
        self.current_frame = value

        if not self.resource_is_loaded():
            return

        self.set_next_item_by_id(self.current_frame)
        (grabbed, frame) = self.get_next_item()

        # TODO do something better if nothing gets grabbed
        if not grabbed:
            return

        self.update_frame(frame)
        self.update_detail(frame)

    def update_detail(self, frame: np.ndarray) -> None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            rect, shape = self.features[self.current_frame][0]
        except IndexError:
            rect, shape = None, None

        self.plotter.widget_detail.set_labels(
            self.closed_eye_left[self.current_frame],
            self.closed_eye_right[self.current_frame],
        )
        self.plotter.widget_detail.frame_d.set_image(np.zeros((20, 20)))
        self.plotter.widget_detail.frame_l.set_image(np.zeros((20, 20)))
        self.plotter.widget_detail.frame_r.set_image(np.zeros((20, 20)))

        if rect is not None or shape is not None:
            eye_left = shape[self.eye_blinking_classifier.eye_left_slice]
            eye_right = shape[self.eye_blinking_classifier.eye_right_slice]
            # get the outer region of the plot
            eye_left_mean = np.nanmean(eye_left, axis=0).astype(np.int32)
            eye_right_mean = np.nanmean(eye_right, axis=0).astype(np.int32)

            self.draw_shape(frame, eye_left, color=(0, 0, 255))
            self.draw_shape(frame, eye_right, color=(255, 0, 0))

            eye_left_width = (
                np.nanmax(eye_left, axis=0)[0] - np.nanmin(eye_left, axis=0)[0]
            ) // 2
            eye_right_width = (
                np.nanmax(eye_right, axis=0)[0] - np.nanmin(eye_right, axis=0)[0]
            ) // 2

            bbox_eye_left = scale_bbox(
                bbox=dlib.rectangle(
                    eye_left_mean[0] - eye_left_width,
                    eye_left_mean[1] - eye_left_width,
                    eye_left_mean[0] + eye_left_width,
                    eye_left_mean[1] + eye_left_width,
                ),
                scale=1,
                padding=-1,
            )
            bbox_eye_right = scale_bbox(
                bbox=dlib.rectangle(
                    eye_right_mean[0] - eye_right_width,
                    eye_right_mean[1] - eye_right_width,
                    eye_right_mean[0] + eye_right_width,
                    eye_right_mean[1] + eye_right_width,
                ),
                scale=1,
                padding=-1,
            )
            img_eye_left = frame[
                bbox_eye_left.top() : bbox_eye_left.bottom(),
                bbox_eye_left.left() : bbox_eye_left.right(),
            ]
            img_eye_right = frame[
                bbox_eye_right.top() : bbox_eye_right.bottom(),
                bbox_eye_right.left() : bbox_eye_right.right(),
            ]
            frame = frame[rect.top() : rect.bottom(), rect.left() : rect.right()]

            self.plotter.widget_detail.frame_d.set_image(frame)
            self.plotter.widget_detail.frame_l.set_image(img_eye_left)
            self.plotter.widget_detail.frame_r.set_image(img_eye_right)

    def save_results(self):
        resource_path = self.get_resource_path()
        result_path = resource_path.parent / (resource_path.stem + ".csv")

        with open(result_path, "w") as f:
            f.write(self.results_file_header)
            for i in range(len(self.score_eye_right)):
                # fancy String literal concatenation
                line = (
                    f"{'closed' if self.closed_eye_left[i] else 'open'};"
                    f"{'closed' if self.closed_eye_right[i] else 'open'};"
                    f"{self.score_eye_left[i]};"
                    f"{self.score_eye_right[i]};"
                    f"{self.valid[i]}"
                    f"\n"
                )
                f.write(line)

    def draw_shape(self, img: np.ndarray, shape: np.ndarray, color: Tuple) -> None:
        for (x, y) in shape:
            cv2.circle(img, (x, y), 1, color, -1)
