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
        # self.setMouseEnabled(x=True, y=True)

    def set_image(self, image: np.ndarray) -> None:
        self.frame.setImage(image)
