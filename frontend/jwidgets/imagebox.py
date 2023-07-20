__all__ = ["JImageBox"]

import numpy as np
import pyqtgraph as pg

class JImageBox(pg.ViewBox):
    def __init__(self, **kwargs):
        super().__init__(invertY=True, lockAspect=True, **kwargs)
        self.frame = pg.ImageItem()
        self.addItem(self.frame)

    def set_image(self, image: np.ndarray) -> None:
        self.frame.setImage(image)