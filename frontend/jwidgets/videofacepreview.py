__all__ = ["JVideoFacePreview"]

from pathlib import Path

import cv2
import numpy as np
import pyqtgraph as pg
import qtawesome as qta
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget
from structlog import get_logger

from frontend.jwidgets.imagebox import JImageBox

logger = get_logger()


class FaceVideoContainer:
    def __init__(self) -> None:
        self.file_path: Path | None = None
        self.resource: cv2.VideoCapture | None = None  # TODO decord as alternative useful?
        self.frame_count: int | None = None

        ccc_file = Path(__file__).parent / "models" / "haarcascade_frontalface_default.xml"
        self.face_finder = cv2.CascadeClassifier(str(ccc_file))

    def load_file(self, file_path: Path) -> np.ndarray:
        assert file_path.exists()
        assert file_path.is_file()

        self.resource = cv2.VideoCapture(str(file_path))
        self.frame_count = int(self.resource.get(cv2.CAP_PROP_FRAME_COUNT))

        return self.get_frame(0)

    def in_range(self, frame_index: int) -> bool:
        assert self.frame_count is not None
        return frame_index >= 0 and frame_index < self.frame_count

    def get_frame(self, frame_index: int) -> np.ndarray:
        assert self.resource is not None

        self.resource.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.resource.read()
        if not ret:
            logger.error("Could not read frame", frame_index=frame_index, file_path=self.file_path)
            return np.zeros((300, 300, 3), dtype=np.uint8)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.face_finder.detectMultiScale(frame_g, 1.1, 5)
        if len(faces) == 0:
            logger.warning("Could not find face", frame_index=frame_index, file_path=self.file_path)
            return frame

        x, y, w, h = faces[0]
        # # scale 0.25 in the y direction
        # y = max(0, y - int(h * 0.25))
        # h = int(h * 1.5)

        x = 0 if x < 0 else x
        y = 0 if y < 0 else y
        w = frame.shape[1] - x if x + w > frame.shape[1] else w
        h = frame.shape[0] - y if y + h > frame.shape[0] else h

        frame = frame[y : y + h, x : x + w]
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
        # expand the widget to the full size of the parent
        self.setMinimumSize(320, 320)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.layout().setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # create an empty label containing a video symbol
        # it contains a video symbol to indicate that here a video frame will be displayed
        self.icon_dragdrop = QLabel()
        self.icon_dragdrop.setPixmap(qta.icon("ri.drag-drop-line", color="gray").pixmap(100, 100))
        self.icon_dragdrop.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.text_dragdrop = QLabel("Drag & Drop\naccording video file here")
        self.text_dragdrop.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.text_dragdrop.setStyleSheet("color: gray;")

        self.layout().addWidget(self.icon_dragdrop)
        self.layout().addWidget(self.text_dragdrop)

        self.glayout = pg.GraphicsLayoutWidget()
        self.face_widget = JImageBox(enableMouse=False, enableMenu=False)
        self.glayout.addItem(self.face_widget)

        self.face_container = FaceVideoContainer()
        self.warn_face_container_not_loaded = False

    def load_file(self, file_path: Path):
        # first do the relayouting
        if self.icon_dragdrop is not None and self.text_dragdrop is not None:
            # this can happen if the user drags & drops a file multiple times
            self.layout().removeWidget(self.icon_dragdrop)
            self.icon_dragdrop.deleteLater()
            self.icon_dragdrop = None
            self.layout().removeWidget(self.text_dragdrop)
            self.text_dragdrop.deleteLater()
            self.text_dragdrop = None

            self.layout().addWidget(self.glayout)

        # then load the file
        frame = self.face_container.load_file(file_path)
        self.face_widget.set_image(frame)

    def set_frame(self, frame_idx: int) -> None:
        assert self.face_container is not None
        if not self.face_container.is_loaded():
            if not self.warn_face_container_not_loaded:
                self.warn_face_container_not_loaded = True
                logger.warning("Face container is not loaded", widget=self)
            return

        if not self.face_container.in_range(frame_idx):
            logger.error("Invalid frame index", frame_idx=frame_idx, frame_count=self.face_container.frame_count)
            return

        frame = self.face_container.get_frame(frame_idx)
        self.face_widget.set_image(frame)

    def dragEnterEvent(self, event: QtGui.QDropEvent):
        logger.info("User started dragging event", widget=self)
        if event.mimeData().hasUrls():
            event.accept()
            logger.info("User started dragging event with mime file", widget=self)
        else:
            event.ignore()
            logger.info("User started dragging event with invalid mime file", widget=self)

    def dropEvent(self, event: QtGui.QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls()]

        if len(files) > 1:
            logger.info("User dropped multiple files", widget=self)
        file = files[0]

        file = Path(file)
        if file.suffix.lower() not in [".mp4", ".flv", ".ts", ".mts", ".avi", ".mov"]:
            logger.info("User dropped invalid file", widget=self)
            return

        self.load_file(file)

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.drawRoundedRect(0, 0, self.width() - 1, self.height() - 1, 10, 10)
        super().paintEvent(a0)
