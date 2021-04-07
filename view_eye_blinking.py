import numpy as np
import dlib
import cv2

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

class view_eye_blinking(QWidget):
    def __init__(self):
        super().__init__()

        layout = QGridLayout()
        self.setLayout(layout)

        self.current_image = None

        self.live_view_button = QRadioButton("Live View")
        self.live_view_button.toggled.connect(self.check_view_mode)
        self.video_view_button = QRadioButton("Video View")
        layout.addWidget(self.live_view_button)
        layout.addWidget(self.video_view_button)

        # video view
        self.disply_width = 640
        self.display_height = 480
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        # create a text label

        # create a vertical box layout and add the two labels
        layout.addWidget(self.image_label)
        # set the vbox layout as the widgets layout


    def check_view_mode(self):
        if self.live_view_button.isChecked() == True:
            print('Start Live View ...')
            # create the video capture thread
            self.thread = VideoThread()
            # connect its signal to the update_image slot
            self.thread.change_pixmap_signal.connect(self.update_live_image)
            # start the thread
            self.thread.start()

        elif self.live_view_button.isChecked() == False:
            print('Stop Live View ...')
            self.thread.quit()
            self.thread.exit()

    @pyqtSlot(np.ndarray)
    def update_live_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
        self.current_image = cv_img

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(0)
        while True:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
