__all__ = ["ImageBox", "SimpleImage"]

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

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


class ImageBox(pg.ViewBox):
    def __init__(
        self,
        face_box: SimpleImage,
        **kwargs,
    ):
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

    def set_selection_image(self, image: np.ndarray) -> None:
        # TODO check if necessary later on...
        self.set_image(image)

        h, w = image.shape[:2]
        # create the position and size of the roi, such that 
        # the roi center is in the middle of the image, and the size 
        # is a rect that would fit a face
        pos = (w / 2 - w / 6, h / 2 - h / 3)
        size = (w / 3, h / 2)
        self.set_roi(pos, size)
        self.set_interactive(True)

        # TODO replace with correct qt import
        self.roi.maxBounds = QtCore.QRectF(0, 0, w, h)

    def set_image(self, image: np.ndarray) -> None:
        self.image = image
        self.frame.setImage(image)
        self.__update(None)

    def set_roi(self, pos: tuple, size: tuple) -> None:
        assert self.roi is not None
        self.roi.setPos(pos)
        self.roi.setSize(size)

    def __update(self, _) -> None:
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