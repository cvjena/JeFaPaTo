__all__ = ["ImageBox"]

import numpy as np
import pyqtgraph as pg


class ImageBox(pg.ViewBox):
    def __init__(
        self,
        parent=None,
        border=None,
        lockAspect=True,
        enableMouse=True,
        invertY=True,
        enableMenu=True,
        name=None,
        invertX=False,
        defaultPadding=0.02,
    ):
        super().__init__(
            parent=parent,
            border=border,
            lockAspect=lockAspect,
            enableMouse=enableMouse,
            invertY=invertY,
            enableMenu=enableMenu,
            name=name,
            invertX=invertX,
            defaultPadding=defaultPadding,
        )

        self.frame = pg.ImageItem()
        self.addItem(self.frame)
        self.roi: pg.ROI | None = None


    def create_roi(self) -> None:
        if self.roi is not None:
            return

        self.roi = pg.ROI(pos=(0, 0), movable=True, resizable=True, rotatable=False, removable=False)
        ## handles scaling horizontally around center
        self.roi.addScaleHandle([1, 0.5], [0.5, 0.5])
        self.roi.addScaleHandle([0, 0.5], [0.5, 0.5])

        ## handles scaling vertically from opposite edge
        self.roi.addScaleHandle([0.5, 0], [0.5, 1])
        self.roi.addScaleHandle([0.5, 1], [0.5, 0])

        ## handles scaling both vertically and horizontally
        self.roi.addScaleHandle([1, 1], [0, 0])
        self.roi.addScaleHandle([0, 0], [1, 1])

        self.addItem(self.roi)

    def set_selection_image(self, image: np.ndarray) -> None:
        self.frame.setImage(image)
        h, w = image.shape[:2]
        
        self.create_roi()

    def set_image(self, image: np.ndarray) -> None:
        self.frame.setImage(image)

    def set_roi(self, pos: tuple, size: tuple) -> None:
        assert self.roi is not None
        self.roi.setPos(pos)
        self.roi.setSize(size)

