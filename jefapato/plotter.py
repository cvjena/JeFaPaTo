from typing import Callable, List

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget


class EyeBlinkingPlotter(QWidget):

    singalFrameChanged = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()

        self._threshold: float = 0.20
        self._current_frame: int = 0

        self.widget_frame: pg.GraphicsLayoutWidget = pg.GraphicsLayoutWidget(
            title="Video Frame"
        )
        self.widget_detail: pg.GraphicsLayoutWidget = pg.GraphicsLayoutWidget()
        self.layout_eye_left: pg.GraphicsLayout = pg.GraphicsLayout()
        self.layout_eye_right: pg.GraphicsLayout = pg.GraphicsLayout()

        # label
        self.label_eye_left: pg.LabelItem = pg.LabelItem("open")
        self.label_eye_right: pg.LabelItem = pg.LabelItem("open")

        image_settings = {
            "invertY": True,
            "lockAspect": True,
            "enableMouse": False,
        }

        self.view_frame: pg.ViewBox = pg.ViewBox(**image_settings)
        self.view_face: pg.ViewBox = pg.ViewBox(**image_settings)
        self.view_image_left: pg.ViewBox() = pg.ViewBox(**image_settings)
        self.view_image_right: pg.ViewBox() = pg.ViewBox(**image_settings)
        # images
        self.image_frame: pg.ImageItem = pg.ImageItem()
        self.image_face: pg.ImageItem = pg.ImageItem()
        self.image_eye_left: pg.ImageItem = pg.ImageItem()
        self.image_eye_right: pg.ImageItem = pg.ImageItem()

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

        # view the frame
        self.widget_frame.addItem(self.view_frame)

        # add the sliders to the plot
        self.widget_graph.addItem(self.indicator_frame)
        self.widget_graph.addItem(self.indicator_threshold)

        # add the images to the image holders
        self.view_frame.addItem(self.image_frame)
        self.view_face.addItem(self.image_face)
        self.view_image_left.addItem(self.image_eye_left)
        self.view_image_right.addItem(self.image_eye_right)

        # setup the detail layout
        self.widget_detail.addItem(self.view_face, row=0, col=0, rowspan=2, colspan=2)
        self.widget_detail.addItem(self.layout_eye_left, row=2, col=1)
        self.widget_detail.addItem(self.layout_eye_right, row=2, col=0)

        # detail right eye
        self.layout_eye_right.addLabel(text="Right eye", row=0, col=0)
        self.layout_eye_right.addItem(self.view_image_right, row=1, col=0)
        self.layout_eye_right.addItem(self.label_eye_right, row=2, col=0)

        # detail left eye
        self.layout_eye_left.addLabel(text="Left eye", row=0, col=0)
        self.layout_eye_left.addItem(self.view_image_left, row=1, col=0)
        self.layout_eye_left.addItem(self.label_eye_left, row=2, col=0)

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

    def set_frame(self, frame: np.ndarray, bgr: bool = False) -> None:
        if bgr:
            frame = frame[..., ::-1]

        self.image_frame.setImage(frame)

    def set_ear_scores(self, left: List[float], right: List[float]) -> None:
        self.curve_left_eye.setData(left)
        self.curve_right_eye.setData(right)

    def set_labels(self, left: bool, right: bool) -> None:
        self.label_eye_left.setText(self.closed_text(left))
        self.label_eye_right.setText(self.closed_text(right))

    def closed_text(self, value: bool) -> str:
        return "closed" if value else "open"

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
