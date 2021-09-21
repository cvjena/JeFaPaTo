from typing import Callable, List, Tuple

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


class EyeBlinkingPlotter(QWidget):

    singalFrameChanged = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()

        self._threshold: float = 0.20
        self._current_frame: int = 0

        self.widget_frame = FrameWidget()
        self.widget_detail = DetailWidget()

        # plotting
        self.widget_graph: pg.PlotWidget = pg.PlotWidget()
        self.widget_graph.setTitle("EAR Score")
        self.widget_graph.setMouseEnabled(x=True, y=False)
        self.widget_graph.setLimits(xMin=0)
        self.widget_graph.setYRange(0, 0.5)
        self.widget_graph.disableAutoRange()
        self.curve_left_eye: pg.PlotDataItem = self.widget_graph.plot()
        self.curve_right_eye: pg.PlotDataItem = self.widget_graph.plot()
        self.curve_left_eye.setPen(pg.mkPen(pg.mkColor(0, 0, 255), width=2))
        self.curve_right_eye.setPen(pg.mkPen(pg.mkColor(255, 0, 0), width=2))

        self.grid_item: pg.GridItem = pg.GridItem()
        self.grid_item.setTickSpacing(x=[1.0], y=[0.1])
        self.widget_graph.addItem(self.grid_item)

        bar_pen = pg.mkPen(width=2, color="k")
        self.indicator_frame: pg.InfiniteLine = pg.InfiniteLine(
            0, movable=False, pen=bar_pen
        )
        self.indicator_threshold: pg.InfiniteLine = pg.InfiniteLine(
            0.2, angle=0, movable=True, pen=bar_pen
        )

        # add the sliders to the plot
        self.widget_graph.addItem(self.indicator_frame)
        self.widget_graph.addItem(self.indicator_threshold)

        self.connect_on_graph_clicked(self.move_line)

    def connect_on_graph_clicked(self, func: Callable) -> None:
        self.widget_graph.scene().sigMouseClicked.connect(func)

    def connect_ruler_dragged(self, func: Callable) -> None:
        self.indicator_frame.sigDragged.connect(func)

    def connect_threshold_dragged(self, func: Callable) -> None:
        self.indicator_threshold.sigDragged.connect(func)

    def move_line(self, mouseClickEvent):
        # this code  calculates the index of the underlying data entry
        # and moves the indicator to it
        vb = self.widget_graph.getPlotItem().vb
        mousePoint = vb.mapSceneToView(mouseClickEvent._scenePos)
        if self.widget_graph.sceneBoundingRect().contains(mouseClickEvent._scenePos):
            mousePoint = vb.mapSceneToView(mouseClickEvent._scenePos)
            index = int(mousePoint.x())
            self.indicator_frame.setPos(index)
            self.singalFrameChanged.emit()

    def enable(self) -> None:
        self.indicator_frame.setMovable(True)
        self.indicator_threshold.setMovable(True)

    def disable(self) -> None:
        self.indicator_frame.setMovable(False)
        self.indicator_threshold.setMovable(False)

    def set_threshold(self, value: float) -> None:
        self._threshold = value
        self.indicator_threshold.setPos(self._threshold)

    def get_threshold(self) -> float:
        return self._threshold

    def update_limits(self, right_border) -> None:
        self.widget_graph.setLimits(xMin=right_border - 100, xMax=right_border)

    def enable_mouse(self, x: bool, y: bool) -> None:
        self.widget_graph.setMouseEnabled(x, y)

    def start(self, fps: float = 30.0) -> None:
        self.widget_graph.enableAutoRange(axis="x")
        self.widget_graph.setMouseEnabled(x=False, y=False)
        self.grid_item.setTickSpacing(x=[int(fps)], y=None)

    def finish(self, last_frame: int):
        self.widget_graph.setLimits(xMin=0, xMax=last_frame)
        self.widget_graph.setXRange(last_frame - 100, last_frame)
        self.widget_graph.setMouseEnabled(x=True, y=False)

    def set_ear_scores(self, left: List[float], right: List[float]) -> None:
        self.curve_left_eye.setData(left)
        self.curve_right_eye.setData(right)
