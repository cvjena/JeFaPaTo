import argparse
import pathlib
import sys

import structlog
from PyQt6.QtWidgets import QApplication

import frontend

logger = structlog.get_logger()

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-tab", type=int, default=0)
    args = parser.parse_args(argv[1:])

    # TODO this is temporary and check if it can be handled somewhere else
    conf = pathlib.Path(__file__).parent / "__static__"
    conf.mkdir(parents=True, exist_ok=True)
    conf_f = conf / "conf.json"
    if not conf_f.exists():
        conf_f.touch(exist_ok=True)
        conf_f.write_text("{}")
    app = QApplication(argv)
    app.setApplicationDisplayName("JeFaPaTo")
    app.setApplicationName("JeFaPaTo")
    splash = frontend.StartUpSplashScreen()
    splash.startup(app)

    ex = frontend.JeFaPaTo(args)

    logger.info("Start JeFaPaTo", version=ex.VERSION)

    ex.show()
    splash.finish(ex)
    sys.exit(app.exec_()) # type: ignore


if __name__ == "__main__":
    main(sys.argv)
