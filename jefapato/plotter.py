from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget

IMAGE_SETTINGS = {
    "invertY": True,
    "lockAspect": True,
    "enableMouse": False,
}


class FrameWidget(pg.GraphicsLayoutWidget):
    def __init__(self, parent=None, show=False, size=None, title=None, **kargs):
        super().__init__(parent=parent, show=show, size=size, title=title, **kargs)

        self.vb = FrameViewBox(**IMAGE_SETTINGS)
        self.addItem(self.vb)
        self.frame = self.vb


class FrameViewBox(pg.ViewBox):
    def __init__(
        self,
        parent=None,
        border=None,
        lockAspect=False,
        enableMouse=True,
        invertY=False,
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

    def set_image(self, image: np.ndarray, bgr: bool = False) -> None:
        if bgr:
            image = image[..., ::-1]

        self.frame.setImage(image)

    def set_image_draw(
        self, image: np.ndarray, shape: np.ndarray, color: Tuple, bgr: bool = False
    ) -> None:
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, color, -1)
        self.set_image(image, bgr)

    def set_image_draw_connected(
        self, image: np.ndarray, shape: np.ndarray, color: Tuple, bgr: bool = False
    ) -> None:
        # TODO
        self.set_image(image, bgr)


class DetailWidget(pg.GraphicsLayoutWidget):
    def __init__(self, parent=None, show=False, size=None, title=None, **kargs):
        super().__init__(parent=parent, show=show, size=size, title=title, **kargs)

        self.layout_l = pg.GraphicsLayout()
        self.layout_r = pg.GraphicsLayout()

        self.frame_d = FrameViewBox(**IMAGE_SETTINGS)
        self.frame_l = FrameViewBox(**IMAGE_SETTINGS)
        self.frame_r = FrameViewBox(**IMAGE_SETTINGS)

        # label
        self.label_l: pg.LabelItem = pg.LabelItem("open")
        self.label_r: pg.LabelItem = pg.LabelItem("open")

        self.addItem(self.frame_d, row=0, col=0, rowspan=2, colspan=2)
        self.addItem(self.layout_l, row=2, col=1)
        self.addItem(self.layout_r, row=2, col=0)

        # detail right eye
        self.layout_r.addLabel(text="Right eye", row=0, col=0)
        self.layout_r.addItem(self.frame_r, row=1, col=0)
        self.layout_r.addItem(self.label_r, row=2, col=0)

        # detail left eye
        self.layout_l.addLabel(text="Left eye", row=0, col=0)
        self.layout_l.addItem(self.frame_l, row=1, col=0)
        self.layout_l.addItem(self.label_l, row=2, col=0)

    def set_labels(self, left: bool, right: bool) -> None:
        self.label_l.setText(self.closed_text(left))
        self.label_r.setText(self.closed_text(right))

    def closed_text(self, value: bool) -> str:
        return "closed" if value else "open"


class GraphWidget(pg.PlotWidget):

    signal_x_ruler_changed = pyqtSignal(float)
    signal_y_ruler_changed = pyqtSignal(float)
    signal_graph_clicked = pyqtSignal(float)

    def __init__(self, parent=None, background="default", plotItem=None, **kargs):
        super().__init__(
            parent=parent, background=background, plotItem=plotItem, **kargs
        )

        self._x_ruler_val: float = 0.0
        self._y_ruler_val: float = 0.2
        self._grid_spacing: float = 30.0
        self._grid_range: float = 100.0

        # self.setTitle("EAR Score")
        self.setMouseEnabled(x=True, y=False)
        self.setLimits(xMin=0)
        self.setYRange(0, 0.5)
        self.disableAutoRange()

        self.grid_item: pg.GridItem = pg.GridItem()
        self.grid_item.setTickSpacing(x=[1.0], y=[0.1])
        self.addItem(self.grid_item)

        bar_pen = pg.mkPen(width=2, color="k")
        self._x_ruler: pg.InfiniteLine = pg.InfiniteLine(
            self._x_ruler_val, movable=False, pen=bar_pen
        )
        self._y_ruler: pg.InfiniteLine = pg.InfiniteLine(
            self._y_ruler_val, angle=0, movable=True, pen=bar_pen
        )

        self._x_ruler.sigDragged.connect(
            lambda _: self.signal_x_ruler_changed.emit(self._x_ruler.getPos()[0])
        )
        self._y_ruler.sigDragged.connect(
            lambda _: self.signal_y_ruler_changed.emit(self._y_ruler.getPos()[1])
        )

        # add the sliders to the plot
        self.addItem(self._x_ruler)
        self.addItem(self._y_ruler)

        self.scene().sigMouseClicked.connect(
            lambda ev: self.signal_graph_clicked.emit(self.move_line(ev))
        )

        self.curves: List[pg.PlotDataItem] = list()

    def move_line(self, mouseClickEvent) -> Optional[float]:
        # this code  calculates the index of the underlying data entry
        # and moves the indicator to it
        vb = self.getPlotItem().vb
        mousePoint = vb.mapSceneToView(mouseClickEvent._scenePos)
        if self.sceneBoundingRect().contains(mouseClickEvent._scenePos):
            mousePoint = vb.mapSceneToView(mouseClickEvent._scenePos)
            self.set_x_ruler(mousePoint.x())
            return self._x_ruler_val

        return None

    def add_curve(self, settings: Dict[str, Any]) -> pg.PlotDataItem:
        curve: pg.PlotDataItem = self.plot()
        curve.setPen(pg.mkPen(settings))

        self.curves.append(curve)
        return curve

    def enable(self) -> None:
        self._x_ruler.setMovable(True)
        self._y_ruler.setMovable(True)

    def disable(self) -> None:
        self._x_ruler.setMovable(False)
        self._y_ruler.setMovable(False)

    def set_x_ruler(self, value: float) -> None:
        self._x_ruler_val = value
        self._x_ruler.setPos(self._x_ruler_val)

    def set_y_ruler(self, value: float) -> None:
        self._y_ruler_val = value
        self._y_ruler.setPos(self._y_ruler_val)

    def start(self, fps: float = 30.0) -> None:
        self.enableAutoRange(axis="x")
        self.setMouseEnabled(x=False, y=False)
        self.set_grid_range(fps)
        self.disable()

    def update(self, location: float) -> None:
        self.setLimits(xMin=location - self._grid_range, xMax=location)
        self.set_x_ruler(location)

    def finish(self, location: float):
        self.setLimits(xMin=0, xMax=location)
        self.setXRange(location - self._grid_range, location)
        self.setMouseEnabled(x=True, y=False)
        self.enable()

    def set_grid_range(self, value: float) -> None:
        self._grid_range = value
        self.grid_item.setTickSpacing(x=[int(self._grid_spacing)], y=None)


class EyeBlinkingPlotter(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self.widget_frame = FrameWidget()
        self.widget_detail = DetailWidget()
        self.widget_graph = GraphWidget()
