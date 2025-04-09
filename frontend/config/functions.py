__all__ = ["Config"]
import json
import os
from pathlib import Path

import pyqtconfig
import structlog

from PyQt6.QtWidgets import QGroupBox


logger = structlog.get_logger()


def load_config_folder() -> Path:
    """
    Load the config path but respect the underlying OS.

    Returns:
        Path: The path to the config file.
    """
    base_path = os.environ.get("APPDATA") or os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    base_path = Path(base_path) / "JeFaPaTo"

    if not os.path.exists(base_path):
        base_path.mkdir(parents=True, exist_ok=True)

    return base_path


def _get_QGroupBox(self):
    return self.isChecked()


def _set_QGroupBox(self, val):
    self.setChecked(val)


def _event_QGroupBox(self):
    return self.clicked


class Config(pyqtconfig.ConfigManager):
    def __init__(self, prefix: str, *args, **kwargs):
        self.prefix = prefix
        base_path = load_config_folder()
        filename = base_path / f"JeFaPaTo_{prefix}.json"
        # if the file does not exist, we create it
        if not filename.exists():
            json.dumps(filename.write_text("{}"), indent=4)

        # check if the file is parseable
        try:
            json.loads(filename.read_text())
        except json.JSONDecodeError:
            logger.error("Config file is not parseable", path=filename)
            # reset to empty file
            json.dumps(filename.write_text("{}"), indent=4)

        super().__init__(*args, filename=filename, **kwargs)

        self.add_hooks(QGroupBox, (_get_QGroupBox, _set_QGroupBox, _event_QGroupBox))

    def geti(self, key: str, default: int = 0) -> int:
        val = self.get(key)
        if val is None:
            return default
        return int(val)

    def getf(self, key: str, default: float = 0.0) -> float:
        val = self.get(key)
        if val is None:
            return default
        return float(val)

    def getb(self, key: str, default: bool = False) -> bool:
        val = self.get(key)
        if val is None:
            return default
        return val

    def gets(self, key: str, default: str = "") -> str:
        val = self.get(key)
        if val is None:
            return default
        return val
