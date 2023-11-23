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

TEXT_DROP_HERE = "Drag and drop a video file here, or click the button in the menu bar."

class JVideoFaceSelection(QWidget):
    """
    Widget for selecting a face region in a video frame.

    Attributes:
        selection_box (pg.ViewBox): ViewBox for displaying the video frame and face selection.
        frame (pg.ImageItem): ImageItem for displaying the video frame.
        face_box (JImageBox): JImageBox for displaying the selected face region.
        roi (pg.ROI): ROI (Region of Interest) for selecting the face region.
        image (np.ndarray | None): The video frame image.
        cb_auto_find (QCheckBox): Checkbox for enabling automatic face detection.
        __handles (dict[str, tuple[tuple[float, float], tuple[float, float]]]): Dictionary of handle positions for the ROI.
        graphics_layout_widget (pg.GraphicsLayoutWidget): GraphicsLayoutWidget for organizing the selection_box and face_box.
        label (QLabel): QLabel for displaying an icon when no video frame is set.
        label_text (QLabel): QLabel for displaying a text when no video frame is set.
    """
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
        self.roi.sigRegionChanged.connect(self.__update_image_roi)

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
        
        self.label = QLabel()
        self.label.setPixmap(qta.icon("ri.drag-drop-line", color="gray").pixmap(100, 100))
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.label_text = QLabel(TEXT_DROP_HERE)
        self.label_text.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_text.setStyleSheet("font-size: 15px;")
        
        self.layout().addWidget(self.label)        
        self.layout().addWidget(self.label_text)
        
        self.face_cascade = cv2.CascadeClassifier(str(Path(__file__).parent / "models" / "haarcascade_frontalface_default.xml"))

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.drawRoundedRect(0, 0, self.width()-1, self.height()-1, 10, 10)
        super().paintEvent(a0)

    def set_selection_image(self, image: np.ndarray) -> None:
        """
        Sets the selection image and updates the widget accordingly.

        Parameters:
            image (np.ndarray): The image to be set as the selection image.

        Returns:
        None
        """    
        if self.image is None:
            try:
                # remove the label
                self.layout().removeWidget(self.label)
                self.layout().removeWidget(self.label_text)
                self.label.deleteLater()
                self.label_text.deleteLater()
                self.label = None
                self.label_text = None
            except AttributeError:
                pass
            # add the graphics layout widget
            self.layout().addWidget(self.graphics_layout_widget)

        self.__set_initial_roi(*self.__auto_find(image))
        self.update_image(image)
        self.set_interactive(True)
        
    def update_image(self, image: np.ndarray) -> None:
        """
        Update the image displayed in the widget.

        Args:
            image (np.ndarray): The image to be displayed.

        Returns:
            None
        """
        self.image = image
        self.frame.setImage(image)
        self.__update_image_roi()
        
    def get_roi_rect(self) -> tuple[int, int, int, int]:
        """
        Returns the coordinates of the region of interest (ROI) rectangle.

        Returns:
            A tuple containing the top, bottom, left, and right coordinates of the ROI rectangle.
        """
        pos = self.roi.pos()
        size = self.roi.size()
        return int(pos.y()), int(pos.y() + size.y()), int(pos.x()), int(pos.x() + size.x())
    
    def set_interactive(self, state: bool) -> None:
        """
        Sets the interactive state of the video face selection widget.

        Parameters:
            state (bool): The interactive state to set. True for interactive, False for non-interactive.
        """
        self.roi.translatable = state
        self.roi.resizable = state

        for key in self.__handles.keys():
            self.__remove_handle(key)
            if state:
                self.__add_handle(key)

    def __set_initial_roi(self, w:int, h: int, pos: tuple, size: tuple) -> None:
        """
        Set the initial region of interest (ROI) for face selection.

        Args:
            w (int): Width of the ROI.
            h (int): Height of the ROI.
            pos (tuple): Position of the ROI (x, y).
            size (tuple): Size of the ROI (width, height).

        Returns:
            None
        """
        self.roi.maxBounds = QRectF(0, 0, w, h)
        self.roi.setPos(pos)
        self.roi.setSize(size)

    def __update_image_roi(self) -> None:
        """
        Update the region of interest (ROI) in the image.

        This method extracts the ROI from the image based on the current
        selection rectangle and updates the face box widget with the
        extracted ROI.

        Returns:
            None
        """
        if self.image is None:
            return        
        y1, y2, x1, x2 = self.get_roi_rect()
        sub_img = self.image[y1:y2, x1:x2]
        self.face_box.set_image(sub_img)

    def __remove_handle(self, handle: str) -> None:
        """
        Removes a handle from the ROI.

        Args:
            handle (str): The name of the handle to be removed.

        Returns:
            None
        """
        handle_attr = getattr(self, handle, None)
        if handle_attr is None:
            logger.warning("ROI Handle does not exist", handle=handle)
            return
        try:
            self.roi.removeHandle(handle_attr)
            delattr(self, handle)
        except IndexError:
            # logger.error("ROI Handle is not attached to ROI", handle=handle)
            pass

    def __add_handle(self, handle: str) -> None:
        """
        Adds a handle to the video face selection widget.

        Args:
            handle (str): The handle to be added.

        Returns:
            None
        """
        if handle in self.__handles.keys():
            pos, center = self.__handles[handle]
            setattr(self, handle, self.roi.addScaleHandle(pos, center))
        else:
            logger.error("Handle does not exist", handle=handle)

    def __fall_back_settings(self, img_w: int, img_h: int) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        Calculate the fallback position and size for face selection.

        Args:
            img_w (int): The width of the image.
            img_h (int): The height of the image.

        Returns:
            tuple[tuple[int, int], tuple[int, int]]: A tuple containing the position and size of the fallback settings.
        """
        pos  = (img_w // 2 - img_w // 6, img_h // 2 - img_h // 3)
        size = (img_w // 3, img_h // 2)
        return pos, size
    
    def __auto_find(self, image: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        Automatically finds the face in the given image and returns the coordinates of the face bounding box.

        Args:
            image (np.ndarray): The input image.

        Returns:
            tuple[tuple[int, int], tuple[int, int]]: A tuple containing the width and height of the image, 
            and the coordinates of the top-left corner and the width and height of the face bounding box.
        """
        
        img_h, img_w = image.shape[:2]
        if not self.cb_auto_find.isChecked():
            return img_w, img_h, *self.__fall_back_settings(img_w, img_h)
        
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(img_gray, 1.3, 5)
        if len(faces) == 0:
            return img_w, img_h, *self.__fall_back_settings(img_w, img_h)

        x, y, w, h = faces[0]
        # increase the size of the box by 50%
        x -= w // 4
        y -= h // 4
        w += w // 2
        h += h // 2
        
        # make sure the box is not out of bounds
        x = 0 if x < 0 else x
        y = 0 if y < 0 else y
        w = image.shape[1] - x if x + w > image.shape[1] else w
        h = image.shape[0] - y if y + h > image.shape[0] else h

        return img_w, img_h, (x, y), (w, h)
