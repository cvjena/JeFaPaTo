__all__ = ["JVideoFacePreview"]

from pathlib import Path

import cv2
import numpy as np
import pyqtgraph as pg
import qtawesome as qta
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget, QHBoxLayout
from structlog import get_logger

from frontend.jwidgets.imagebox import JImageBox
from jefapato.facial_features import MediapipeFaceModel

logger = get_logger()


class FaceVideoContainer:
    def __init__(self) -> None:
        self.file_path: Path | None = None
        self.resource: cv2.VideoCapture | None = None  # TODO decord as alternative useful?
        self.frame_count: int | None = None
        self.frame_current: int = 0
        self.rotation_state = 0

        self.model = MediapipeFaceModel()

    def load_file(self, file_path: Path) -> None:
        if not file_path.exists():
            logger.error("File does not exist", file_path=file_path)
            return
        if not file_path.is_file():
            logger.error("File is not a file", file_path=file_path)
            return

        self.resource = cv2.VideoCapture(str(file_path))
        self.frame_count = int(self.resource.get(cv2.CAP_PROP_FRAME_COUNT))

    def in_range(self, frame_index: int) -> bool:
        if self.frame_count is None:
            return False

        return frame_index >= 0 and frame_index < self.frame_count

    def set_rotate_state(self, rotation_state: int) -> None:
        self.rotation_state = rotation_state

    def get_frame(self, frame_index: int | None = None) -> tuple[np.ndarray, tuple[int, int, int, int], bool]:
        """
        Returns the frame at the given index and the face bounding box.
        """

        def default(frame: np.ndarray | None = None) -> tuple[np.ndarray, tuple[int, int, int, int], bool]:
            if frame is None:
                frame = np.zeros((300, 300, 3), dtype=np.uint8)
            return frame, (0, 0, frame.shape[1], frame.shape[0]), False

        if self.resource is None:
            logger.error("Resource is not loaded", file_path=self.file_path)
            return default()

        self.frame_current = frame_index or self.frame_current

        # TODO here might be a strong offset to the actual frame index if the video is very very long
        self.resource.set(cv2.CAP_PROP_POS_FRAMES, self.frame_current)
        ret, frame = self.resource.read()
        if not ret:
            logger.error("Could not read frame", frame_index=self.frame_current, file_path=self.file_path)
            return default()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 0 -> 0°, 1 -> 90°, 2 -> 180°, 3 -> 270°
        if self.rotation_state == 1:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation_state == 2:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation_state == 3:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        landmarks, _, valid = self.model.extract(frame)

        if not valid:
            logger.warning("Could not find face", frame_index=self.frame_current, file_path=self.file_path)
            return default(frame)

        bbox = self.model.lmk_to_bbox(landmarks)
        return frame, bbox, True

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

        # track the rotation state the user can select
        # 0 -> 0°, 1 -> 90°, 2 -> 180°, 3 -> 270°

        rot_wid = QWidget()
        rot_hbox = QHBoxLayout()

        self.rotation_state = 0
        # two buttons and a label to display the current rotation state
        self.rotation_label = QLabel("Rotation: 0°")
        self.rotation_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.rotation_label.setStyleSheet("color: gray;")
        self.rotation_label.setFixedHeight(30)
        self.rotation_label.setFixedWidth(100)

        self.status_info = QLabel("Status: No file loaded")
        self.status_info.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.status_info.setStyleSheet("color: gray;")
        self.status_info.setFixedHeight(30)
        self.status_info.setFixedWidth(150)

        self.rotation_button_l = QtWidgets.QPushButton(icon=qta.icon("ph.arrow-arc-left", color="gray"))  # type: ignore
        self.rotation_button_r = QtWidgets.QPushButton(icon=qta.icon("ph.arrow-arc-right", color="gray"))  # type: ignore
        self.rotation_button_l.clicked.connect(self.rotate_left)
        self.rotation_button_r.clicked.connect(self.rotate_right)

        rot_hbox.addWidget(self.rotation_button_l)
        rot_hbox.addWidget(self.rotation_label)
        rot_hbox.addWidget(self.rotation_button_r)
        rot_hbox.addWidget(self.status_info)

        rot_wid.setLayout(rot_hbox)
        self.vbox.addWidget(rot_wid)

        self.setLayout(self.vbox)

    def rotate_left(self):
        self.rotation_state = (self.rotation_state - 1) % 4
        self.rotation_label.setText(f"Rotation: {self.rotation_state * 90}°")
        self.face_container.set_rotate_state(self.rotation_state)
        frame, bbox, valid = self.face_container.get_frame()
        self.face_widget.set_image_with_bbox(frame, bbox)
        self.update_status(valid)

    def rotate_right(self):
        self.rotation_state = (self.rotation_state + 1) % 4
        self.rotation_label.setText(f"Rotation: {self.rotation_state * 90}°")
        self.face_container.set_rotate_state(self.rotation_state)
        frame, bbox, valid = self.face_container.get_frame()
        self.face_widget.set_image_with_bbox(frame, bbox)
        self.update_status(valid)

    def update_status(self, face_found: bool):
        if face_found:
            self.status_info.setText("Status: Face found")
            self.status_info.setStyleSheet("color: green;")
        else:
            self.status_info.setText("Status: No face found")
            self.status_info.setStyleSheet("color: red;")
        self.status_info.update()

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

            self.vbox.insertWidget(0, self.glayout)

        # then load the file
        self.face_container.load_file(file_path)
        self.set_frame(0)

    def set_frame(self, frame_idx: int) -> None:
        if self.face_container is None:
            logger.error("Face container is not set", widget=self)
            return

        if not self.face_container.is_loaded():
            if not self.warn_face_container_not_loaded:
                self.warn_face_container_not_loaded = True
                logger.warning("Face container is not loaded", widget=self)
            return

        if not self.face_container.in_range(frame_idx):
            logger.error("Invalid frame index", frame_idx=frame_idx, frame_count=self.face_container.frame_count)
            return

        frame, bbox, valid = self.face_container.get_frame(frame_idx)
        self.face_widget.set_image_with_bbox(frame, bbox)
        self.update_status(valid)

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
