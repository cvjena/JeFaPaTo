__all__ = ["JGraph"]

from typing import Any

import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget, QHBoxLayout

class JGraph(QWidget):
    signal_x_ruler_changed = pyqtSignal(float)
    signal_y_ruler_changed = pyqtSignal(float)
    signal_graph_clicked   = pyqtSignal(float)

    def __init__(self, background="default", x_lim_max=1000, **kargs):
        super().__init__()
        self.plot_item = pg.PlotItem(background=background, **kargs)
                
        self.plot_item.setLimits(xMin=0, xMax=x_lim_max)
        self.plot_item.setXRange(0, x_lim_max)
        self.plot_item.setYRange(0, 1)

        self.axis_b: pg.AxisItem = self.plot_item.getAxis("bottom")
        self.curves: list[pg.PlotDataItem] = []
        self.scatters: list[pg.ScatterPlotItem] = []

        self.setLayout(QHBoxLayout())
        
        gl = pg.GraphicsLayoutWidget()
        gl.addItem(self.plot_item)
        self.layout().addWidget(gl)

    def add_curve(self, *args, **kwargs) -> pg.PlotDataItem:
        curve: pg.PlotDataItem = self.plot_item.plot(*args, **kwargs)
        curve.setPen(pg.mkPen(*args, **kwargs))

        self.curves.append(curve)
        return curve

    def remove_curve(self, curve: pg.PlotDataItem) -> None:
        self.plot_item.removeItem(curve)
        self.curves.remove(curve)

    # TODO check if setting is needed
    def add_scatter(self, settings: dict[str, Any] | None = None) -> pg.ScatterPlotItem:
        scatter: pg.ScatterPlotItem = pg.ScatterPlotItem()
        self.plot_item.addItem(scatter)
        self.scatters.append(scatter)
        return scatter
    
    def getViewBox(self) -> pg.ViewBox:
        return self.plot_item.getViewBox()
    
    def setXRange(self, min: float, max: float) -> None:
        self.plot_item.setXRange(min, max)

    def setYRange(self, min: float, max: float) -> None:
        self.plot_item.setYRange(min, max)

    def setLimits(self, xMin: float=0, xMax: float=0, yMin: float=0, yMax: float=0) -> None:
        self.plot_item.setLimits(xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax)

    def getAxis(self, name: str) -> pg.AxisItem:
        return self.plot_item.getAxis(name)
    
    def removeItem(self, item: pg.GraphicsObject) -> None:  
        self.plot_item.removeItem(item)

    def clear(self) -> None:
        self.plot_item.clear()
        self.curves.clear()
        self.scatters.clear()