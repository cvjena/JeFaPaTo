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
from jefapato.facial_features import MediapipeFaceModel

logger = get_logger()


class FaceVideoContainer:
    def __init__(self) -> None:
        self.file_path: Path | None = None
        self.resource: cv2.VideoCapture | None = None  # TODO decord as alternative useful?
        self.frame_count: int | None = None

        self.model = MediapipeFaceModel()

    def load_file(self, file_path: Path) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        assert file_path.exists()
        assert file_path.is_file()

        self.resource = cv2.VideoCapture(str(file_path))
        self.frame_count = int(self.resource.get(cv2.CAP_PROP_FRAME_COUNT))

        return self.get_frame(0)

    def in_range(self, frame_index: int) -> bool:
        assert self.frame_count is not None
        return frame_index >= 0 and frame_index < self.frame_count

    def get_frame(self, frame_index: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """
        Returns the frame at the given index and the face bounding box.
        """

        assert self.resource is not None

        # TODO here might be a strong offset to the actual frame index if the video is very very long
        self.resource.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.resource.read()
        if not ret:
            logger.error("Could not read frame", frame_index=frame_index, file_path=self.file_path)
            return np.zeros((300, 300, 3), dtype=np.uint8), (0, 0, 300, 300)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks, _, valid = self.model.extract(frame)

        if not valid:
            logger.warning("Could not find face", frame_index=frame_index, file_path=self.file_path)
            return frame, (0, 0, frame.shape[1], frame.shape[0])

        bbox = self.model.lmk_to_bbox(landmarks)
        return frame, bbox

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
        self.vbox = QVBoxLayout()

        self.setAcceptDrops(True)
        # expand the widget to the full size of the parent
        self.setMinimumSize(320, 320)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.vbox.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # create an empty label containing a video symbol
        # it contains a video symbol to indicate that here a video frame will be displayed
        self.icon_dragdrop = QLabel()
        self.icon_dragdrop.setPixmap(qta.icon("ri.drag-drop-line", color="gray").pixmap(100, 100))
        self.icon_dragdrop.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.text_dragdrop = QLabel("Drag & Drop\naccording video file here")
        self.text_dragdrop.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.text_dragdrop.setStyleSheet("color: gray;")

        self.vbox.addWidget(self.icon_dragdrop)
        self.vbox.addWidget(self.text_dragdrop)

        self.glayout = pg.GraphicsLayoutWidget()
        self.face_widget = JImageBox(interactive=True, enableMouse=True, enableMenu=False)
        self.glayout.addItem(self.face_widget)

        self.face_container = FaceVideoContainer()
        self.warn_face_container_not_loaded = False

        self.setLayout(self.vbox)

    def load_file(self, file_path: Path):
        # first do the relayouting
        if self.icon_dragdrop is not None and self.text_dragdrop is not None:
            # this can happen if the user drags & drops a file multiple times
            self.vbox.removeWidget(self.icon_dragdrop)
            self.icon_dragdrop.deleteLater()
            self.icon_dragdrop = None
            self.vbox.removeWidget(self.text_dragdrop)
            self.text_dragdrop.deleteLater()
            self.text_dragdrop = None

            self.vbox.addWidget(self.glayout)

        # then load the file
        frame, bbox = self.face_container.load_file(file_path)
        self.face_widget.set_image_with_bbox(frame, bbox)

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

        frame, bbox = self.face_container.get_frame(frame_idx)
        self.face_widget.set_image_with_bbox(frame, bbox)

    def dragEnterEvent(self, event: QtGui.QDropEvent):  # type: ignore
        logger.info("User started dragging event", widget=self)

        if (mime_data := event.mimeData()) is None:
            return

        if mime_data.hasUrls():
            event.accept()
            logger.info("User started dragging event with mime file", widget=self)
        else:
            event.ignore()
            logger.info("User started dragging event with invalid mime file", widget=self)

    def dropEvent(self, event: QtGui.QDropEvent):  # type: ignore
        if (mime_data := event.mimeData()) is None:
            return

        files = [u.toLocalFile() for u in mime_data.urls()]

        if len(files) > 1:
            logger.info("User dropped multiple files", widget=self)
        file = files[0]

        file = Path(file)
        if file.suffix.lower() not in [".mp4", ".flv", ".ts", ".mts", ".avi", ".mov"]:
            logger.info("User dropped invalid file", widget=self)
            return

        self.load_file(file)

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:  # type: ignore
        painter = QtGui.QPainter(self)
        painter.drawRoundedRect(0, 0, self.width() - 1, self.height() - 1, 10, 10)
        super().paintEvent(a0)
