__all__ = ["load", "save"]

import json
import pathlib

import pyqtconfig
import structlog

from .defaults import DEFAULTS

__config_path = pathlib.Path(__file__).parent.parent.parent / "__static__" / "conf.json"
logger = structlog.get_logger()


def load() -> pyqtconfig.ConfigManager:

    if not __config_path.exists():
        logger.info("Config file not found", path=__config_path)
        __config_path.write_text(json.dumps(DEFAULTS))

    # we load the config file
    data = json.loads(__config_path.read_text())

    # check if all keys in the config file are in the defaults
    for key in data:
        if key not in DEFAULTS:
            logger.warning("Key not found in defaults. Add it.", key=key)
            data[key] = DEFAULTS[key]

    return pyqtconfig.ConfigManager(data)


def save(config: pyqtconfig.ConfigManager):
    __config_path.write_text(json.dumps(config.as_dict()))
    logger.info("Config file written", path=__config_path)
