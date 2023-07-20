__all__ = ["JGraph"]

from typing import Any

import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal

class JGraph(pg.PlotItem):
    signal_x_ruler_changed = pyqtSignal(float)
    signal_y_ruler_changed = pyqtSignal(float)
    signal_graph_clicked   = pyqtSignal(float)

    def __init__(self, parent=None, background="default", x_lim_max=1000, **kargs):
        super().__init__(parent=parent, background=background, **kargs)

        self.axis_b: pg.AxisItem = self.getAxis("bottom")
        self.setLimits(xMin=0, xMax=x_lim_max)
        self.setXRange(0, x_lim_max)
        self.setYRange(0, 1)

        self.curves: list[pg.PlotDataItem] = []
        self.scatters: list[pg.ScatterPlotItem] = []

    def add_curve(self, *args, **kwargs) -> pg.PlotDataItem:
        curve: pg.PlotDataItem = self.plot(*args, **kwargs)
        curve.setPen(pg.mkPen(*args, **kwargs))

        self.curves.append(curve)
        return curve

    def remove_curve(self, curve: pg.PlotDataItem) -> None:
        self.removeItem(curve)
        self.curves.remove(curve)

    # TODO check if setting is needed
    def add_scatter(self, settings: dict[str, Any] | None = None) -> pg.ScatterPlotItem:
        scatter: pg.ScatterPlotItem = pg.ScatterPlotItem()
        self.addItem(scatter)
        self.scatters.append(scatter)
        return scatter