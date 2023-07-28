__all__ = ["JVideoFaceSelection"]

from pathlib import Path

import cv2
import qtawesome as qta
import numpy as np
import pyqtgraph as pg
import structlog
from qtpy.QtWidgets import QCheckBox, QWidget, QVBoxLayout, QLabel
from qtpy.QtCore import Qt, QRectF
from PyQt6 import QtCore, QtGui

from frontend.jwidgets.imagebox import JImageBox 

logger = structlog.get_logger(__name__)

PEN = pg.mkPen(color="g", width=3, style=Qt.PenStyle.DashLine, join=Qt.PenJoinStyle.RoundJoin, cap=Qt.PenCapStyle.RoundCap)
PEN_H = pg.mkPen(color="r", width=3,  style=Qt.PenStyle.DashLine, join=Qt.PenJoinStyle.RoundJoin, cap=Qt.PenCapStyle.RoundCap)
PEN_HANDLE = pg.mkPen(color="k", width=8, style=Qt.PenStyle.SolidLine, join=Qt.PenJoinStyle.RoundJoin, cap=Qt.PenCapStyle.RoundCap)

class JVideoFaceSelection(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.selection_box: pg.ViewBox = pg.ViewBox(invertY=True, lockAspect=True, enableMenu=False, enableMouse=False)
        self.frame = pg.ImageItem()
        self.selection_box.addItem(self.frame)
    
        self.face_box: JImageBox = JImageBox(enableMouse=False, enableMenu=False)
        self.roi: pg.ROI = pg.ROI(
            pos=(0, 0), 
            movable=True, 
            resizable=True, 
            rotatable=False, 
            removable=False, 
            pen=PEN, 
            handlePen=PEN_HANDLE, 
            hoverPen=PEN_H, 
            handleHoverPen=PEN_H
        )
        ## handles scaling horizontally around center
        self.selection_box.addItem(self.roi)
        self.roi.sigRegionChanged.connect(self.__update)

        self.image: np.ndarray | None = None
        self.cb_auto_find = QCheckBox("Auto find face")

        self.__handles: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
            "h1" : ((1.0, 0.5), (0.5, 0.5)),
            "h2" : ((0.0, 0.5), (0.5, 0.5)),
            "h3" : ((0.5, 0.0), (0.5, 1.0)),
            "h4" : ((0.5, 1.0), (0.5, 0.0)),
            "h5" : ((1.0, 1.0), (0.0, 0.0)),
            "h6" : ((0.0, 0.0), (1.0, 1.0)),
        }
        self.set_interactive(False)
        self.graphics_layout_widget = pg.GraphicsLayoutWidget()
        # set the margins to 10px
        self.graphics_layout_widget.ci.setSpacing(10)
        self.graphics_layout_widget.addItem(self.selection_box)
        self.graphics_layout_widget.addItem(self.face_box)
        
        self.setLayout(QVBoxLayout())
        self.layout().setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        # self.layout().addWidget(graphics_layout_widget)        
        
        self.label = QLabel()
        self.label.setPixmap(qta.icon("ri.drag-drop-line", color="gray").pixmap(100, 100))
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.label_text = QLabel("Drag and drop a video file here, or click the button, or start the Webcam")
        self.label_text.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_text.setStyleSheet("font-size: 15px;")
        
        self.layout().addWidget(self.label)        
        self.layout().addWidget(self.label_text)

    def set_selection_image(self, image: np.ndarray) -> None:
        assert self.label is not None, "Label must be set before selection image can be set"
        assert self.label_text is not None, "Label text must be set before selection image can be set"
        # remove the label
        self.layout().removeWidget(self.label)
        self.layout().removeWidget(self.label_text)
        self.label.deleteLater()
        self.label_text.deleteLater()
        self.label = None
        self.label_text = None

        # add the graphics layout widget
        self.layout().addWidget(self.graphics_layout_widget)

        self.set_image(image)
        self.set_roi(*self.__auto_find())
        self.set_interactive(True)

    def set_image(self, image: np.ndarray) -> None:
        self.image = image
        self.frame.setImage(image)
        h, w = image.shape[:2]
        self.roi.maxBounds = QRectF(0, 0, w, h)
        self.__update()

    def set_roi(self, pos: tuple, size: tuple) -> None:
        self.roi.setPos(pos)
        self.roi.setSize(size)

    def __update(self) -> None:
        if self.image is None:
            return        
        y1, y2, x1, x2 = self.get_roi_rect()
        sub_img = self.image[y1:y2, x1:x2]
        self.face_box.set_image(sub_img)

    def get_roi_rect(self) -> tuple[int, int, int, int]: 
        pos = self.roi.pos()
        size = self.roi.size()
        return int(pos.y()), int(pos.y() + size.y()), int(pos.x()), int(pos.x() + size.x())

    def set_interactive(self, state: bool) -> None:
        self.roi.translatable = state
        self.roi.resizable = state

        for key in self.__handles.keys():
            self.__remove_handle(key)
            if state:
                self.__add_handle(key)

    def __remove_handle(self, handle: str) -> None:
        handle_attr = getattr(self, handle, None)
        if handle_attr is None:
            logger.warning("ROI Handle does not exists", handle=handle)
            return
        try:
            self.roi.removeHandle(handle_attr)
            delattr(self, handle)
        except IndexError:
            logger.error("ROI Handle is not attached to ROI", handle=handle)
            pass

    def __add_handle(self, handle: str) -> None:
        if handle in self.__handles.keys():
            pos, center = self.__handles[handle]
            setattr(self, handle, self.roi.addScaleHandle(pos, center))
        else:
            logger.error("Handle does not exists", handle=handle)

    def __fall_back_settings(self) -> tuple[tuple[int, int], tuple[int, int]]:
        assert self.image is not None, "Image must be set before fall back settings can be used"
        h, w = self.image.shape[:2]
        pos = (w // 2 - w // 6, h // 2 - h // 3)
        size = (w // 3, h // 2)
        return pos, size
    

    def __auto_find(self) -> tuple[tuple[int, int], tuple[int, int]]:
        assert self.image is not None, "Image must be set before auto find can be used"

        if not self.cb_auto_find.isChecked():
            return self.__fall_back_settings()
        
        path = Path(__file__).parent / "models" / "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(str(path))
        img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)
        if len(faces) == 0:
            return self.__fall_back_settings()

        x, y, w, h = faces[0]
        # increase the size of the box by 50%
        x -= w // 4
        y -= h // 4
        w += w // 2
        h += h // 2
        return (x, y), (w, h)
    
    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.drawRoundedRect(0, 0, self.width()-1, self.height()-1, 10, 10)
        super().paintEvent(a0)