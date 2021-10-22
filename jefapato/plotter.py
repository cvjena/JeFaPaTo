from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import dlib
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSignal

IMAGE_SETTINGS = {
    "invertY": True,
    "lockAspect": True,
    "enableMouse": False,
}


@dataclass
class BoundingBox:
    l: int = 0
    r: int = -1
    t: int = 0
    b: int = -1


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
        self.setMouseEnabled(x=True, y=True)

    def set_image(
        self,
        image: np.ndarray,
        bbox: BoundingBox = BoundingBox(),
        bgr: bool = False,
    ) -> None:
        if bgr:
            image = image[..., ::-1]

        image = image[bbox.t : bbox.b, bbox.l : bbox.r]
        self.frame.setImage(image)


class EyeDetailWidget(pg.GraphicsLayoutWidget):
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

    def set_frame(
        self,
        frame: np.ndarray,
        rect: dlib.rectangle,
        shape: np.ndarray,
        eye_padding: float = 1.5,
    ) -> None:
        self.frame_d.set_image(np.zeros((20, 20)))
        self.frame_l.set_image(np.zeros((20, 20)))
        self.frame_r.set_image(np.zeros((20, 20)))

        if rect is not None or shape is not None:
            eye_l = shape[slice(42, 48)]
            eye_r = shape[slice(36, 42)]

            eye_l_m = np.nanmean(eye_l, axis=0).astype(np.int32)
            eye_r_m = np.nanmean(eye_r, axis=0).astype(np.int32)

            eye_l_w = (np.nanmax(eye_l, axis=0)[0] - np.nanmin(eye_l, axis=0)[0]) // 2
            eye_r_w = (np.nanmax(eye_r, axis=0)[0] - np.nanmin(eye_r, axis=0)[0]) // 2
            eye_l_w = int(eye_padding * eye_l_w)
            eye_r_w = int(eye_padding * eye_r_w)

            crop = eye_l_w if eye_l_w > eye_r_w else eye_r_w

            bbox_l = BoundingBox(
                l=eye_l_m[0] - crop,
                r=eye_l_m[0] + crop,
                t=eye_l_m[1] - crop,
                b=eye_l_m[1] + crop,
            )
            bbox_r = BoundingBox(
                l=eye_r_m[0] - crop,
                r=eye_r_m[0] + crop,
                t=eye_r_m[1] - crop,
                b=eye_r_m[1] + crop,
            )

            bbox_f = BoundingBox(
                l=bbox_r.l,
                r=bbox_l.r,
                t=min(bbox_l.t, bbox_r.t),
                b=max(bbox_l.b, bbox_r.b),
            )

            self.draw_points(frame, eye_l, (0, 0, 255))
            self.draw_points(frame, eye_r, (255, 0, 0))

            self.draw_ratio(frame, eye_l, (0, 0, 255))
            self.draw_ratio(frame, eye_r, (255, 0, 0))

            self.frame_d.set_image(frame, bbox_f)
            self.frame_l.set_image(frame, bbox_l)
            self.frame_r.set_image(frame, bbox_r)

    def closed_text(self, value: bool) -> str:
        return "closed" if value else "open"

    def draw_points(self, frame, shape, color):
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, color, -1)

    def draw_ratio(self, frame, shape, color):
        # horizontal line
        cv2.line(
            frame, (shape[0][0], shape[0][1]), (shape[3][0], shape[3][1]), color, 1
        )

        # vertical line
        t = (shape[1] + shape[2]) // 2
        b = (shape[4] + shape[5]) // 2
        cv2.line(frame, (t[0], t[1]), (b[0], b[1]), color, 1)


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

        self.axis_b: pg.AxisItem = self.getAxis("bottom")
        self.axis_b.setTickSpacing(300, self._grid_spacing)

        # self.setTitle("EAR Score")
        self.setMouseEnabled(x=True, y=False)
        self.setLimits(xMin=0)
        self.setYRange(0, 0.5)
        self.disableAutoRange()

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
        self.scatters: List[pg.ScatterPlotItem] = list()

    def add_grid(self) -> None:
        self.grid_item: pg.GridItem = pg.GridItem()
        self.grid_item.setTickSpacing(x=[300], y=[0.1])
        self.addItem(self.grid_item)

    def remove_grid(self) -> None:
        try:
            self.removeItem(self.grid_item)
        except AttributeError:
            pass

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

    def add_scatter(self, settigns: Dict[str, Any] = None) -> pg.ScatterPlotItem:
        scatter: pg.ScatterPlotItem = pg.ScatterPlotItem()
        self.addItem(scatter)
        self.scatters.append(scatter)
        return scatter

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
        self.set_grid_spacing(fps)
        self.disable()

    def update(self, location: float) -> None:
        self.setXRange(location - self._grid_range, location)
        self.setLimits(xMin=location - self._grid_range, xMax=location)
        self.set_x_ruler(location)

    def finish(self, location: float):
        self.setLimits(xMin=0, xMax=location)
        self.setXRange(location - self._grid_range, location)
        self.setMouseEnabled(x=True, y=False)
        self.enable()

    def set_grid_spacing(self, value: float) -> None:
        self._grid_spacing = value
