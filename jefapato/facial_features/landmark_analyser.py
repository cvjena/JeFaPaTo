__all__ = ["FaceAnalyzer"]

from typing import Any, Type
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import pluggy
import psutil
import structlog

from .features import Feature, FeatureData
from .video_data_loader import VideoDataLoader
from .mediapipe_landmark_extractor import MediapipeLandmarkExtractor
from .queue_items import AnalyzeQueueItem

logger = structlog.get_logger()

class FaceAnalyzer():
    hookimpl = pluggy.HookimplMarker("analyser")
    hookspec = pluggy.HookspecMarker("analyser")
    
    def __init__(self, max_ram_size: int = 4<<28):
        """
        Initialize the LandmarkAnalyser object.

        Args:
            max_ram_size (int): Maximum RAM size in bytes. Defaults to 4<<28.
        """
        
        super().__init__()
        self.max_ram_size = max_ram_size

        self.feature_classes: OrderedDict[str, Feature] = OrderedDict()
        self.feature_data: OrderedDict[str, list[FeatureData]] = OrderedDict()

        self.resource_interface: Any = None
        self.video_resource: Path | int | None = None
        self.data_amount: int = 0

        self.pm = pluggy.PluginManager("analyser")

    def analysis_setup(self, bbox_slice: tuple[int, int, int, int] | None = None) -> bool:
        """
        Sets up the analysis by initializing necessary components and calculating available resources.

        Args:
            bbox_slice (tuple[int, int, int, int] | None): Optional bounding box slice.

        Returns:
            bool: True if the setup is successful.

        Raises:
            ValueError: If no resource interface is set.
        """
        if self.resource_interface is None:
            logger.error("No resource interface set.")
            raise ValueError("No resource interface set.")

        # compute how much RAM space we have
        available_memory = min(psutil.virtual_memory().available, self.max_ram_size)
        item_size = self.get_item_size(in_bytes=True)
        items_to_place = int(available_memory // item_size)

        w, h, d = self.get_item_size()
        logger.info("Item information", width=w, height=h, depth=d)
        logger.info("Available memory", memory=available_memory)
        logger.info("Data loader queue space", space=items_to_place)

        self.loader = VideoDataLoader(self.resource_interface.read, data_amount=self.data_amount, queue_maxsize=items_to_place)
        self.extractor = MediapipeLandmarkExtractor(data_queue=self.loader.data_queue, data_amount=self.data_amount, bbox_slice=bbox_slice)
        self.extractor.register(self)
        return True

    def analysis_start(self):
        """
        Starts the analysis process by initializing the loader and extractor.
        
        Raises:
            RuntimeError: If the loader or extractor is not set up.
        """
        
        if not hasattr(self, "loader") or not hasattr(self, "extractor"):
            logger.error("Loader or extractor not set up.")
            raise RuntimeError("Loader or extractor not set up.")
        
        self.loader.start()
        logger.info("Started loader thread.", loader=self.loader)

        self.extractor.start()
        logger.info("Started extractor thread.", extractor=self.extractor)
        
        # only trigger the started hook if there are any registered plugins
        if len(self.pm.get_plugins()) > 0:
            self.pm.hook.started()

    def stop(self):
        """
        Stops the loader and extractor threads if they exist.
        """
        if (loader := getattr(self, "loader", None)) is not None:
            loader.stopped = True
            loader.join()
        if (extractor := getattr(self, "extractor", None)) is not None:
            extractor.stopped = True
            extractor.join()

    def register_hooks(self, plugin: object) -> None:
        """
        Register hooks for the given plugin.
        Gives access to the following hooks:
            - started
            - finished
            - paused
            - resumed
            - processed_percentage
            - updated_display
            - updated_feature
        Args:
            plugin (object): The plugin object to register hooks for.

        Returns:
            None
        """
        try:
            self.pm.register(plugin)
        except ValueError:
            pass

    def set_features(self, features: list[Type[Feature]]) -> None:
        """
        Sets the features for the landmark analyser.
        
        The features are stored in a dictionary with the feature name as key and the feature object as value.
        The features are automatically tracked over time and stored in a list for saving it later.
        
        If the list of features is empty, the feature classes and data are cleared nevertheless.

        Args:
            features (list[Type[Feature]]): A list of feature classes.

        Returns:
            None
        """
        # raise runtime error if the extractor is already running
        if hasattr(self, "extractor") and self.extractor.isRunning():
            raise RuntimeError("Cannot set features while the extractor is running.")
  
        if features is None:
            logger.error("Features cannot be None.")
            raise ValueError("Features cannot be None.")
    
        if not isinstance(features, list):
            # if it is not a list, we make it a list to make it easier to handle, just like a single feature
            features = [features]
        
        self.feature_classes.clear()
        self.feature_data.clear()

        for feature in features:
            if feature is None:
                logger.error("Feature cannot be None.")
                raise ValueError("Feature cannot be None.")
            
            if not isinstance(feature, type):
                logger.error("Feature is not a class.", feature=feature)
                raise ValueError(f"Feature {feature} is not a class.")
            
            if not issubclass(feature, Feature):
                logger.error("Feature is not a subclass of Feature.", feature=feature)
                raise ValueError(f"Feature {feature} is not a subclass of Feature.")
            
            self.feature_classes[feature.__name__] = feature()
            self.feature_data[feature.__name__] = []

    def toggle_pause(self) -> None:
        """
        Toggles the pause state of the extractor.
        
        This is just a convenience function to make it easier to pause and resume the extractor,
        used in the GUI.
        
        The hooks are triggered in the extractor itself!
        """
        # raise runtime error if the extractor is not running
        if not hasattr(self, "extractor"):
            raise RuntimeError("Cannot toggle pause while the extractor setup.")

        self.extractor.toggle_pause()

    def clean_start(self, bbox_slice: tuple[int, int, int, int] | None = None) -> None:
        """
        Starts the landmark analysis process.
        
        This function is used to start the analysis process.
        It sets up the analysis, starts the loader and extractor threads and connects the hooks.
        The process is stopped by calling the stop() function, or paused by calling the toggle_pause() function.

        Args:
            bbox_slice (tuple[int, int, int, int] | None): Optional bounding box slice.

        Returns:
            None
        """
        for m_name in self.feature_classes:
            self.feature_data[m_name].clear()

        self.analysis_setup(bbox_slice=bbox_slice)
        self.analysis_start()

    def get_header(self) -> list[str]:
        header = ["frame"]
        for feature in self.feature_classes.values():
            header.extend(feature.get_header())
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

        for feature_name, feature_data in self.feature_data.items():
            feature_class = self.feature_classes[feature_name]
            row.extend([x for x in feature_class.as_row(feature_data[self.__iter_counter - 1])])

        return row

    def prepare_video_resource(self, value: Path) -> tuple[bool, np.ndarray]:
        self.video_resource = value

        if not isinstance(self.video_resource, Path):
            raise ValueError("Video resource must be a Path or an integer.")

        if not self.video_resource.exists():
            raise FileNotFoundError(f"File {self.video_resource} does not exist.")
        
        if not self.video_resource.is_file():
            raise ValueError(f"File {self.video_resource} is not a file.")

        if self.video_resource.suffix.lower() not in [".mp4", ".flv", ".ts", ".mts", ".avi", ".mov"]:
            raise ValueError(f"File {self.video_resource} is not a video file.")

        self.resource_interface = cv2.VideoCapture(str(self.video_resource.absolute()))
        self.data_amount = self.resource_interface.get(cv2.CAP_PROP_FRAME_COUNT)
        
        success, image = self.resource_interface.read()
        return success, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def get_fps(self) -> float:
        """
        Get the frames per second (FPS) of the video capture.
        Can be used for internal time calculations to display the correct frame.
        However, this fully depends on the actual video capture, so it might not be accurate.

        Returns:
            float: The frames per second of the video capture.
        """
        return self.resource_interface.get(cv2.CAP_PROP_FPS)

    def get_item_size(self, in_bytes: bool = False) -> tuple[int, int, int] | int:
        """
        Get the size of the item.

        Parameters:
        - in_bytes (bool): Whether to return the size in bytes. Default is False.

        Returns:
        - tuple[int, int, int] | int: If in_bytes is True, returns the size in bytes.
            Otherwise, returns a tuple of width, height, and channels.
        """
        width  = int(self.resource_interface.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.resource_interface.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # set the number of channels to 3, because we are using RGB images
        channels = 3
        if in_bytes:
            return width * height * channels
        return width, height, channels

    def get_throughput(self) -> tuple[int, int]:
        """
        Returns the throughput of the loader and extractor in items per second.

        Returns:
            A tuple containing the processing per second of the loader and extractor.
        """
        if not hasattr(self, "loader") or not hasattr(self, "extractor"):
            raise RuntimeError("Loader or extractor not set up.")

        return self.loader.processing_per_second, self.extractor.processing_per_second
    
    @MediapipeLandmarkExtractor.hookimpl
    def handle_update(self, item: AnalyzeQueueItem) -> None:
        # logger.debug("Handling update")
        # here would be some drawing? and storing of the features we are interested in
        temp_data = OrderedDict()
        image = item.image
        valid = item.valid
        features = item.landmark_features
        blendshapes = item.blendshape_features
        x_offset, y_offset = item.x_offset, item.y_offset

        for f_name, f_class in self.feature_classes.items():
            # if it f_class is a blendshape feature, we need to pass the blendshapes
            if f_class.is_blendshape:
                feature_data = f_class.compute(blendshapes, valid)
            else:
                feature_data = f_class.compute(features, valid)
            f_class.draw(image=image, data=feature_data, x_offset=x_offset, y_offset=y_offset)
            self.feature_data[f_name].append(feature_data)
            temp_data[f_name] = feature_data
            
        # connect the hooks only if there are any plugins registered
        if len(self.pm.get_plugins()) > 0:
            self.pm.hook.updated_feature(feature_data=temp_data)
            self.pm.hook.updated_display(image=image)
            
    @MediapipeLandmarkExtractor.hookimpl
    def update_progress(self, perc: float) -> None:
        """
        Trigger a hook that the progress was updated.
        """
        if len(self.pm.get_plugins()) > 0:
            self.pm.hook.processed_percentage(percentage=perc)

    @MediapipeLandmarkExtractor.hookimpl
    def handle_pause(self) -> None:
        """
        Trigger a hook that the extractor was paused.
        """
        if len(self.pm.get_plugins()) > 0:
            self.pm.hook.paused()
        
    @MediapipeLandmarkExtractor.hookimpl
    def handle_resume(self) -> None:
        if len(self.pm.get_plugins()) > 0:
            self.pm.hook.resumed()

    @MediapipeLandmarkExtractor.hookimpl
    def handle_finished(self) -> None:
        """
        Trigger a hook that the extractor finished.
        """
        if len(self.pm.get_plugins()) > 0:
            self.pm.hook.finished()
    
    @hookspec
    def updated_display(self, image: np.ndarray):
        """
        Trigger a hook that the displaying information was updated.
        """

    @hookspec
    def updated_feature(self, feature_data: OrderedDict[str, Any]) -> None:
        """
        Trigger a hook that the features were updated.
        """
        
    @hookspec
    def processed_percentage(self, percentage: float) -> None:
        """
        Trigger a hook that the percentage was updated.
        """
    
    @hookspec
    def started(self) -> None:
        """
        Trigger a hook that the analysis started.
        """
    
    @hookspec
    def paused(self) -> None:
        """
        Trigger a hook that the analysis was paused.
        """
    
    @hookspec
    def resumed(self) -> None:
        """
        Trigger a hook that the analysis was resumed.
        """
        
    @hookspec
    def finished(self) -> None:
        """
        Trigger a hook that the analysis finished.
        """