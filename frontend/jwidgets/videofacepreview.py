__all__ = ["JVideoFacePreview"]

from pathlib import Path

import cv2
import numpy as np
import pyqtgraph as pg
import qtawesome as qta
from PyQt6 import QtCore
from PyQt6.QtGui import QDropEvent
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget
from structlog import get_logger

from frontend.plotting.image_box import SimpleImage

logger = get_logger()

class FaceVideoContainer:
    def __init__(self) -> None:
        self.file_path: Path | None = None
        self.resource: cv2.VideoCapture | None = None # TODO decord as alternative useful?
        self.frame_count: int | None = None

        ccc_file =  Path(__file__).parent / "models" / "haarcascade_frontalface_default.xml"
        self.face_finder = cv2.CascadeClassifier(str(ccc_file))

    def load_file(self, file_path: Path) -> np.ndarray:
        assert file_path.exists()
        assert file_path.is_file()

        self.resource = cv2.VideoCapture(str(file_path))
        self.frame_count = int(self.resource.get(cv2.CAP_PROP_FRAME_COUNT))
        
        return self.get_frame(0)
    
    def in_range(self, frame_number: int) -> bool:
        assert self.frame_count is not None
        return frame_number >= 0 and frame_number < self.frame_count

    def get_frame(self, frame_number: int) -> np.ndarray:
        assert self.resource is not None

        self.resource.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.resource.read()
        if not ret:
            logger.error("Could not read frame", frame_number=frame_number, file_path=self.file_path)
            return np.zeros((300, 300, 3), dtype=np.uint8)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.face_finder.detectMultiScale(frame_g, 1.1, 5)
        if len(faces) == 0:
            logger.warning("Could not find face", frame_number=frame_number, file_path=self.file_path)
            return frame

        x, y, w, h = faces[0]
        frame = frame[y:y+h, x:x+w]
        return frame
    
    def is_loaded(self) -> bool:
        return self.resource is not None


class JVideoFacePreview(QWidget):
    """
    This class is a Qt widget that displays a video frame with a face.
    
    It contains a line edit which accepts drag and drop to load the file.
    It is also only shown during certain triggers.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setLayout(QVBoxLayout())
        self.setAcceptDrops(True)
        self.setMinimumSize(300, 300)

        # draw round corners
        self.setStyleSheet("border-radius: 10px; border: 1px solid gray;")

        # create an empty label containing a video symbol
        # it contains a video symbol to indicate that here a video frame will be displayed
        self.label = QLabel()
        self.label.setPixmap(qta.icon("ri.drag-drop-line", color="gray").pixmap(100, 100))
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.layout().addWidget(self.label)        

        self.glayout = pg.GraphicsLayoutWidget()
        self.face_widget = SimpleImage(enableMouse=False, enableMenu=False)
        self.glayout.addItem(self.face_widget)

        self.face_container = FaceVideoContainer()

    def load_file(self, file_path: Path):
        # first do the relayouting
        if self.label is not None:
            # this can happen if the user drags & drops a file multiple times
            self.layout().removeWidget(self.label)
            self.label.deleteLater()
            self.label = None
            self.layout().addWidget(self.glayout)

        # then load the file
        frame = self.face_container.load_file(file_path)        
        self.face_widget.set_image(frame)

    def set_frame(self, frame_idx: int) -> None:
        assert self.face_container is not None
        if not self.face_container.is_loaded():
            logger.warning("Face container is not loaded", widget=self)
            return

        if not self.face_container.in_range(frame_idx):
            logger.error("Invalid frame index", frame_idx=frame_idx, frame_count=self.face_container.frame_count)
            return

        frame = self.face_container.get_frame(frame_idx)
        self.face_widget.set_image(frame)

    def dragEnterEvent(self, event: QDropEvent):
        logger.info("User started dragging event", widget=self)
        if event.mimeData().hasUrls():
            event.accept()
            logger.info("User started dragging event with mime file", widget=self)
        else:
            event.ignore()
            logger.info("User started dragging event with invalid mime file", widget=self)

    def dropEvent(self, event: QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls()]

        if len(files) > 1:
            logger.info("User dropped multiple files", widget=self)
        file = files[0]

        file = Path(file)
        if file.suffix.lower() not in [".mp4", ".flv", ".ts", ".mts", ".avi", ".mov"]:
            logger.info("User dropped invalid file", widget=self)
            return
        
        self.load_file(file)