import argparse
import sys

import structlog
from PyQt6.QtWidgets import QApplication

import frontend

logger = structlog.get_logger()

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-tab", type=int, default=0)
    args = parser.parse_args(argv[1:])

    app = QApplication(argv)
    splash = frontend.StartUpSplashScreen()
    splash.startup(app)

    ex = frontend.JeFaPaTo(args)
    logger.info("Start JeFaPaTo")

    ex.show()
    splash.finish(ex)
    sys.exit(app.exec_()) # type: ignore


if __name__ == "__main__":
    main(sys.argv)
