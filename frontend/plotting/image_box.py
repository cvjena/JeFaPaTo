__all__ = ["FaceSelectBox", "SimpleImage"]

from pathlib import Path

import cv2
import numpy as np
import pyqtgraph as pg
import structlog
from pyqtgraph.Qt import QtCore
from qtpy.QtWidgets import QCheckBox

logger = structlog.get_logger(__name__)


PEN = pg.mkPen(color="g", width=3, style=QtCore.Qt.DashLine, join=QtCore.Qt.RoundJoin, cap=QtCore.Qt.RoundCap)
PEN_H = pg.mkPen(color="r", width=3,  style=QtCore.Qt.DashLine, join=QtCore.Qt.RoundJoin, cap=QtCore.Qt.RoundCap)
PEN_HANDLE = pg.mkPen(color="k", width=8, style=QtCore.Qt.SolidLine, join=QtCore.Qt.RoundJoin, cap=QtCore.Qt.RoundCap)

class SimpleImage(pg.ViewBox):
    def __init__(self, **kwargs):
        super().__init__(invertY=True, lockAspect=True, **kwargs)
        self.frame = pg.ImageItem()
        self.addItem(self.frame)

    def set_image(self, image: np.ndarray) -> None:
        self.frame.setImage(image)


class FaceSelectBox(pg.ViewBox):
    def __init__(self, face_box: SimpleImage, **kwargs):
        super().__init__(invertY=True, lockAspect=True, **kwargs)
        self.frame = pg.ImageItem()
        self.addItem(self.frame)
        
        self.face_box: SimpleImage = face_box
        self.roi: pg.ROI = pg.ROI(pos=(0, 0), movable=True, resizable=True, rotatable=False, removable=False, pen=PEN, handlePen=PEN_HANDLE, hoverPen=PEN_H, handleHoverPen=PEN_H)
        ## handles scaling horizontally around center
        self.addItem(self.roi)
        self.roi.sigRegionChanged.connect(self.__update)

        self.image: np.ndarray | None = None
        self.set_interactive(False)

        self.cb_auto_find = QCheckBox("Auto find face")
        self.cb_auto_find.setChecked(True)

    def set_selection_image(self, image: np.ndarray) -> None:
        self.set_image(image)
        self.set_roi(*self.__auto_find())
        self.set_interactive(True)

    def set_image(self, image: np.ndarray) -> None:
        self.image = image
        self.frame.setImage(image)
        h, w = image.shape[:2]
        self.roi.maxBounds = QtCore.QRectF(0, 0, w, h)
        self.__update()

    def set_roi(self, pos: tuple, size: tuple) -> None:
        self.roi.setPos(pos)
        self.roi.setSize(size)

    def __update(self) -> None:
        if self.image is None:
            return        
        y1, y2, x1, x2 = self.get_roi_rect()
        sub_img = self.image[y1:y2, x1:x2]
        self.face_box.set_image(sub_img)

    def get_roi_rect(self) -> tuple[int, int, int, int]: 
        pos = self.roi.pos()
        size = self.roi.size()
        return int(pos.y()), int(pos.y() + size.y()), int(pos.x()), int(pos.x() + size.x())

    def set_interactive(self, state: bool) -> None:
        self.roi.translatable = state
        self.roi.resizable = state

        if state:
            self.roi.addScaleHandle([1, 0.5], [0.5, 0.5])
            self.roi.addScaleHandle([0, 0.5], [0.5, 0.5])

            ## handles scaling vertically from opposite edge
            self.roi.addScaleHandle([0.5, 0], [0.5, 1])
            self.roi.addScaleHandle([0.5, 1], [0.5, 0])

            ## handles scaling both vertically and horizontally
            self.roi.addScaleHandle([1, 1], [0, 0])
            self.roi.addScaleHandle([0, 0], [1, 1])
        else:
            try:
                self.roi.removeHandle(5)
                self.roi.removeHandle(4)
                self.roi.removeHandle(3)
                self.roi.removeHandle(2)
                self.roi.removeHandle(1)
                self.roi.removeHandle(0)
            except IndexError:
                pass


    def __fall_back_settings(self) -> tuple[tuple[int, int], tuple[int, int]]:
        assert self.image is not None, "Image must be set before fall back settings can be used"
        h, w = self.image.shape[:2]
        pos = (w // 2 - w // 6, h // 2 - h // 3)
        size = (w // 3, h // 2)
        return pos, size
    

    def __auto_find(self) -> tuple[tuple[int, int], tuple[int, int]]:
        assert self.image is not None, "Image must be set before auto find can be used"

        if not self.cb_auto_find.isChecked():
            return self.__fall_back_settings()
        
        path = Path(__file__).parent / "models" / "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(str(path))
        img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
        if len(faces) == 0:
            return self.__fall_back_settings()

        x, y, w, h = faces[0]
        # increase the size of the box by 50%
        x -= w // 4
        y -= h // 4
        w += w // 2
        h += h // 2
        return (x, y), (w, h)