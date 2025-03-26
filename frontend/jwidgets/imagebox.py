__all__ = ["JImageBox"]

import numpy as np
import pyqtgraph as pg
from PyQt6.QtGui import QCursor
from PyQt6 import QtCore


class JImageBox(pg.ViewBox):
    def __init__(self, interactive: bool = False, **kwargs):
        super().__init__(invertY=True, lockAspect=True, **kwargs)
        self.frame = pg.ImageItem()
        self.interactive = interactive
        self.addItem(self.frame)
        self.setBackgroundColor("w")

        # if interactive changet the cursor to a cross
        if self.interactive:
            # self.setMouseMode(self.RectMode)
            self.setCursor(QCursor(QtCore.Qt.CursorShape.OpenHandCursor))

        self._bbox_rect = None

    def set_image(self, image: np.ndarray) -> None:
        self.frame.setImage(image)
        self.setLimits(xMin=0, xMax=image.shape[1], yMin=0, yMax=image.shape[0])
        self.setRange(xRange=[0, image.shape[1]], yRange=[0, image.shape[0]], padding=0)

    def set_image_with_bbox(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        self.set_image(image)

        x1, y1, x2, y2 = bbox
        if self._bbox_rect:
            self.removeItem(self._bbox_rect)
        self._bbox_rect = pg.RectROI([x1, y1], [x2 - x1, y2 - y1], pen=pg.mkPen("r", width=4), movable=False)
        self.addItem(self._bbox_rect)

        # zoom to bbox, but autorange is not working
        self.setRange(xRange=[x1, x2], yRange=[y1, y2], padding=0)
