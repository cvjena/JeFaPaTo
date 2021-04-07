import os
import numpy as np
#import dlib
import cv2
import shutil
import glob


from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *

class view_eye_blinking(QWidget):
    def __init__(self, camera_id):
        super().__init__()

        layout = QGridLayout()
        self.setLayout(layout)

        ## PROBERTIES

        self.current_image = None
        self.video_file_path = ''
        self.extract_folder = os.path.join('', 'tmp')
        self.image_paths = []
        #self.camera_id = camera_id

        # ==============================================================================================================

        ## GUI ELEMENTS

        #self.live_view_button = QRadioButton("Live View")
        #self.live_view_button.toggled.connect(self.check_view_mode)
        #self.video_view_button = QRadioButton("Video View")
        #layout.addWidget(self.live_view_button)
        #layout.addWidget(self.video_view_button)

        self.openButton = QPushButton("Open Video")
        self.openButton.setToolTip("Open Video File")
        self.openButton.setStatusTip("Open Video File")
        self.openButton.setFixedHeight(24)
        self.openButton.setIconSize(QSize(16, 16))
        self.openButton.setFont(QFont("Noto Sans", 8))
        self.openButton.setIcon(QIcon.fromTheme("document-open", QIcon("./")))
        self.openButton.clicked.connect(self.load_video)
        layout.addWidget(self.openButton, 0, 0)

        # video view
        self.disply_width = 640
        self.display_height = 480
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
         # create a vertical box layout and add the two labels
        layout.addWidget(self.image_label, 1,0)

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(2, 0)
        self.positionSlider.setToolTip(str(self.positionSlider.value()))
        self.positionSlider.sliderMoved.connect(self.set_position)
        layout.addWidget(self.positionSlider)

        self.label_slider_value = QLabel('0')
        layout.addWidget(self.label_slider_value, 3, 0)

        # PLOTTING
        self.plot_image_label = QLabel(self)
        self.plot_image_label.resize(self.disply_width, self.display_height)
        # create a vertical box layout and add the two labels
        layout.addWidget(self.plot_image_label, 4, 0)

        label_threshold = QLabel('Threshold:')
        layout.addWidget(label_threshold,1,1)

        edit_threshold = QLineEdit('0')
        layout.addWidget(edit_threshold,1 ,2 )


        # ==============================================================================================================

        ## INITIALIZATION ROUTINES


        if os.path.isfile(os.path.join(self.extract_folder,"frame_00000000.png")):
            self.show_image()

            # set slider
            self.positionSlider.setRange(0, len(self.image_paths))

    def set_position(self):
        self.show_image(self.positionSlider.value())


    def load_video(self,):
        print('load video from file')
        fileName, _ = QFileDialog.getOpenFileName(self, "Select video file",
                                                  ".", "Video Files (*.mp4 *.flv *.ts *.mts *.avi)")

        if fileName != '':
            self.video_file_path = fileName
            self.extract_video()

    @pyqtSlot(list)
    def extract_video(self):
        if os.path.isdir(self.extract_folder):
            shutil.rmtree(self.extract_folder, ignore_errors=True)
            os.mkdir(self.extract_folder)

        self.thread = ExtractImagesThread(self.openButton, self.video_file_path, self.image_paths, self.extract_folder)
        self.thread.image_paths_signal.connect(self.show_image)
        self.thread.start()

    def show_image(self, image_id = 0):
        self.image_paths = glob.glob(os.path.join(self.extract_folder, './*.png'))
        self.image_paths.sort()

        self.label_slider_value.setText("Frame Number:\t" + str(image_id))

        cv_img = cv2.imread(self.image_paths[image_id])
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

'''
    def check_view_mode(self):
        if self.live_view_button.isChecked() == True:
            print('Start Live View ...')
            # create the video capture thread
            self.thread = VideoThread(self.camera_id)
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

'''
class ExtractImagesThread (QThread):
    image_paths_signal = pyqtSignal(list)
    def __init__(self, openButton, video_file_path, image_paths, extract_folder):
        super().__init__()
        self.openButton = openButton
        self.video_file_path = video_file_path
        self.image_paths = image_paths
        self.extract_folder = extract_folder

    def run(self):

        vidcap = cv2.VideoCapture(self.video_file_path)
        success, image = vidcap.read()
        count = 0
        while success:
            self.openButton.setText(str(count))
            image_path = os.path.join(self.extract_folder, "frame_%08d.png" % count)
            self.image_paths.append(image_path)
            cv2.imwrite(image_path, image)
            success, image = vidcap.read()
            # print('Read a new frame: ', success)
            count += 1
        self.openButton.setText('Open Video')
        self.image_paths.sort()
        # show images
'''
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    def __init__(self, camera_id):

        super().__init__()
        self.camera_id = camera_id
    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(self.camera_id)
        if not cap:
            print("check camera parameter")
        while True:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
'''