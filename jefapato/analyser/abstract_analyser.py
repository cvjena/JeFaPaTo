__all__ = ["Analyser", "hookimpl", "hookspec"]

import abc
import pathlib
from typing import Any, Callable, List, Optional, Type, Union

import pluggy
import psutil
import structlog

from jefapato import extracting, loading

MAX_RAM_SIZE = 4 << 28  # ~4GB
logger = structlog.get_logger(__name__)
hookspec = pluggy.HookspecMarker("analyser")
hookimpl = pluggy.HookimplMarker("analyser")


class Analyser(abc.ABC):
    def __init__(
        self,
        loader_c: Type[loading.DataLoader],
        extractor_c: Type[extracting.Extractor],
    ) -> None:
        super().__init__()
        self.loader_c = loader_c
        self.extractor_c = extractor_c

        self.features: List[Any] = list()

        self.functions_processing_started: List[Callable] = list()
        self.functions_processing_updated: List[Callable] = list()
        self.functions_processing_finished: List[Callable] = list()
        self.functions_processed_percentage: List[Callable] = list()

        self.resource: Any = None
        self.resource_path: Optional[pathlib.Path] = None
        self.data_amount: int = 0

        self.pm = pluggy.PluginManager("analyser")
        self.pm.add_hookspecs(self)

    def resource_path_is_set(self):
        return isinstance(self.resource_path, pathlib.Path)

    @abc.abstractmethod
    def load_resource(self):
        pass

    @abc.abstractmethod
    def release_resource(self):
        pass

    @abc.abstractmethod
    def get_next_item(self):
        pass

    @abc.abstractmethod
    def set_data_amount(self):
        pass

    @abc.abstractmethod
    def reset_data_resource(self):
        pass

    @abc.abstractmethod
    def set_next_item_by_id(self, value: int):
        pass

    @abc.abstractmethod
    def get_item_size_in_bytes(self):
        pass

    @abc.abstractmethod
    def get_item_size(self):
        pass

    @abc.abstractmethod
    def call_after_resource_load(self, data: Any) -> None:
        pass

    def analysis_setup(self):
        self.features = list()
        self.classes = list()
        # reset the data resource to the initial state, like frame 0
        self.reset_data_resource()

        # compute how much RAM space we have
        available_memory = min(psutil.virtual_memory().available, MAX_RAM_SIZE)
        item_size = self.get_item_size_in_bytes()
        items_to_place = int(available_memory // item_size)

        w, h, d = self.get_item_size()
        b = self.get_item_size_in_bytes()
        logger.info("Item information", width=w, height=h, depth=d, bytes=b)
        logger.info("Available memory", memory=available_memory)
        logger.info("Data loader queue space", space=items_to_place)

        self.loader = self.loader_c(self.get_next_item, items_to_place)
        self.extractor = self.extractor_c(
            data_queue=self.loader.data_queue, data_amount=self.data_amount
        )

        self.extractor.processedPercentage.connect(
            lambda x: self.pm.hook.processed_percentage(percentage=x)
        )
        self.extractor.processingFinished.connect(lambda: self.pm.hook.finished())

    def analysis_start(self):
        self.pm.hook.started()

        self.loader.start()
        logger.info("Started loader thread.", loader=self.loader)
        self.extractor.start()
        logger.info("Started extractor thread.", extractor=self.extractor)

    def current_feature(self):
        return self.features[-1]

    def current_class(self):
        return self.classes[-1]

    def set_resource_path(self, value: Union[str, pathlib.Path]) -> Optional[Any]:
        if isinstance(value, str):
            value = pathlib.Path(value)
        self.resource_path = value
        self.load_resource()
        _, data = self.get_next_item()
        # set to inital index again because we just loaded it
        self.set_next_item_by_id(0)
        self.call_after_resource_load()

    def get_resource_path(self) -> Union[pathlib.Path, None]:
        return self.resource_path

    def get_data_amount(self) -> int:
        return self.data_amount

    def resource_is_loaded(self):
        return self.resource is not None

    def stop(self):
        self.loader.stopped = True
        self.extractor.stopped = True

    def reset(self):
        self.features = list()

    def register_hooks(self, plugin: object) -> None:
        self.pm.register(plugin)

    @hookspec
    def started(self):
        """
        Inform however needs it that the anlysis has started.
        """

    @hookspec
    def finished(self):
        """
        Inform however needs it that the anlysis has finished.
        """

    @hookspec
    def processed_percentage(self, percentage: int):
        """
        Inform however needs it about the percentage done.
        """
