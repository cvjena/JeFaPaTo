__all__ = ["Config"]

import json
import pathlib

import pyqtconfig
import structlog

from .defaults import DEFAULTS

logger = structlog.get_logger()


class Config(pyqtconfig.ConfigManager):
    __config_path = (
        pathlib.Path(__file__).parent.parent.parent / "__static__" / "conf.json"
    )
    __cache = {}

    def __init__(self, prefix: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.prefix = prefix
        self.load_conf()

    def __to_default(self):
        logger.info("Config file not found", path=self.__config_path)
        self.__config_path.write_text(json.dumps(DEFAULTS))

    def __check_prefix_exists(self):
        if self.prefix not in self.__cache:
            self.__cache[self.prefix] = DEFAULTS[self.prefix]
            self.save_conf()

    def load_conf(self) -> pyqtconfig.ConfigManager:
        if not self.__config_path.exists():
            self.__to_default()

        # we load the config file
        self.__cache = json.loads(self.__config_path.read_text())
        self.__check_prefix_exists()
        data = self.__cache[self.prefix]

        # check if all keys in the config file are in the defaults
        for key in data:
            if key not in DEFAULTS[self.prefix]:
                logger.warning(
                    "Key not found in [prefix]. Add it.", key=key, prefix=self.prefix
                )
                data[key] = DEFAULTS[self.prefix][key]

        # we cannot use set_defaults because it would trigger a lot of hooks
        # and we don't want that
        self.defaults = data

    def save_conf(self):
        self.__cache[self.prefix] = self.as_dict()
        self.__config_path.write_text(json.dumps(self.__cache))
        logger.info("Config file written", path=self.__config_path, prefix=self.prefix)
