__all__ = ["WidgetGraph"]

from typing import Any, Dict, List, Optional

import pyqtgraph as pg
from qtpy import QtCore


class WidgetGraph(pg.PlotItem):
    signal_x_ruler_changed = QtCore.Signal(float)
    signal_y_ruler_changed = QtCore.Signal(float)
    signal_graph_clicked = QtCore.Signal(float)

    def __init__(
        self,
        parent=None,
        background="default",
        add_ruler=True,
        add_xruler=True,
        add_yruler=True,
        **kargs
    ):
        super().__init__(parent=parent, background=background, **kargs)

        self._x_ruler_val: float = 0.0
        self._y_ruler_val: float = 0.2
        self._grid_spacing: float = 30.0
        self._grid_range: float = 100.0
        self.add_ruler = add_ruler
        self.add_xruler = add_xruler
        self.add_yruler = add_yruler

        self.axis_b: pg.AxisItem = self.getAxis("bottom")
        self.axis_b.setTickSpacing(300, self._grid_spacing)

        # self.setTitle("EAR Score")
        self.setMouseEnabled(x=True, y=False)
        self.setLimits(xMin=0)
        self.setYRange(0, 0.5)
        self.disableAutoRange()

        bar_pen = pg.mkPen(width=2, color="k")
        if self.add_ruler:
            if self.add_xruler:
                self._x_ruler: pg.InfiniteLine = pg.InfiniteLine(
                    self._x_ruler_val, movable=False, pen=bar_pen
                )
                self._x_ruler.sigDragged.connect(
                    lambda _: self.signal_x_ruler_changed.emit(
                        self._x_ruler.getPos()[0]
                    )
                )
                self.addItem(self._x_ruler)
            if self.add_yruler:
                self._y_ruler: pg.InfiniteLine = pg.InfiniteLine(
                    self._y_ruler_val, angle=0, movable=True, pen=bar_pen
                )
                self._y_ruler.sigDragged.connect(
                    lambda _: self.signal_y_ruler_changed.emit(
                        self._y_ruler.getPos()[1]
                    )
                )
                # add the sliders to the plot
                self.addItem(self._y_ruler)

        # self.scene().sigMouseClicked.connect(
        #     lambda ev: self.signal_graph_clicked.emit(self.move_line(ev))
        # )

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
        if not self.add_ruler:
            return None
        # this code  calculates the index of the underlying data entry
        # and moves the indicator to it
        vb = self.getPlotItem().vb
        mousePoint = vb.mapSceneToView(mouseClickEvent._scenePos)
        if self.sceneBoundingRect().contains(mouseClickEvent._scenePos):
            mousePoint = vb.mapSceneToView(mouseClickEvent._scenePos)
            self.set_x_ruler(mousePoint.x())
            return self._x_ruler_val

        return None

    def add_curve(self, *args, **kwargs) -> pg.PlotDataItem:
        curve: pg.PlotDataItem = self.plot(*args, **kwargs)
        curve.setPen(pg.mkPen(*args, **kwargs))

        self.curves.append(curve)
        return curve

    def remove_curve(self, curve: pg.PlotDataItem) -> None:
        self.removeItem(curve)
        self.curves.remove(curve)

    def add_scatter(self, settings: Dict[str, Any] = None) -> pg.ScatterPlotItem:
        scatter: pg.ScatterPlotItem = pg.ScatterPlotItem()
        self.addItem(scatter)
        self.scatters.append(scatter)
        return scatter

    def enable(self) -> None:
        if self.add_xruler:
            self._x_ruler.setMovable(True)
        if self.add_yruler:
            self._y_ruler.setMovable(True)

    def disable(self) -> None:
        if self.add_xruler:
            self._x_ruler.setMovable(False)
        if self.add_yruler:
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
