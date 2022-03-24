__all__ = ["Analyser", "hookimpl", "hookspec"]

import abc
import pathlib
from typing import Any, Tuple, Type, Union

import pluggy
import psutil
import structlog

from jefapato import extracting, loading

MAX_RAM_SIZE = 4 << 28  # ~4GB
logger = structlog.get_logger()
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

        self.loader = None
        self.extractor = None

        self.resource: Any = None
        self.resource_path: Union[pathlib.Path, int] = None
        self.data_amount: int = 0

        self.pm = pluggy.PluginManager("analyser")
        self.pm.add_hookspecs(self)

    def analysis_setup(self) -> bool:
        if self.resource is None:
            if self.resource_path is None:
                logger.error("Resource could not be created!")
                return False
            self.load_resource()

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

    def get_resource_path(self) -> Union[pathlib.Path, None]:
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

    @abc.abstractmethod
    def load_resource(self):
        pass

    @abc.abstractmethod
    def release_resource(self):
        pass

    @abc.abstractmethod
    def get_next_item(self) -> Tuple[bool, Any]:
        pass

    @abc.abstractmethod
    def get_item_size_in_bytes(self) -> int:
        pass

    @abc.abstractmethod
    def get_item_size(self) -> Tuple[int, int, int]:
        pass

    @hookspec
    def started(self):
        """
        Inform however needs it that the anlysis was started.
        """

    @hookspec
    def paused(self):
        """
        Inform however needs it that the anlysis was paused.
        """

    @hookspec
    def resumed(self):
        """
        Inform however needs it that the anlysis was resumed.
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
