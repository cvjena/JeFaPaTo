import os
import sys
from typing import Mapping
from PyQt5.QtCore import right
from matplotlib.backends.backend_qt5 import ToolCopyToClipboardQT
import numpy as np
import dlib
import cv2
import getopt
import argparse
import glob
import imutils
from imutils import face_utils
from numpy.lib.index_tricks import AxisConcatenator

def scale_bbox(bbox: dlib.rectangle, scale: float, padding: int = 0) -> dlib.rectangle:
    """Scale and pad a dlib.rectangle

    Copyright: Tim Büchner tim.buechner@uni-jena.de

    This function will scale a dlib.rectangle and can also apply additional padding
    to the rectangle. If a padding value of -1 is given, the padding is will be
    a fourth of the respective widht and height.

    Args:
        bbox (dlib.rectangle): rectangle which will be scaled and optionally padded
        scale (float): scaling factor for the rectangle
        padding (int, optional): Additional padding for the rectangle. Defaults to 0. If -1 use fourth of width and height for padding.

    Returns:
        dlib.rectangle: newly scaled (and optionally padded) dlib.rectangle
    """    
    left: int = int(bbox.left() * scale)
    top: int = int(bbox.top() * scale)
    right: int = int(bbox.right() * scale)
    bottom: int = int(bbox.bottom() * scale)

    width = right - left
    height = bottom - top

    padding_w = width // 4 if padding == -1 else padding
    padding_h = height // 4 if padding == -1 else padding

    return dlib.rectangle(
        left=left - padding_w,
        top=top - padding_h,
        right=right + padding_w,
        bottom=bottom + padding_h,
    )

class EyeBlinkingDetector():
    def __init__(self, threshold):
        super().__init__()

        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        self.shape_predictor_file = './data/shape_predictor_68_face_landmarks.dat'
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.shape_predictor_file)

        self.threshold = threshold

        self.left_closed = False
        self.right_closed = False

        self.left_eye_closing_norm_area = -1
        self.right_eye_closing_norm_area = -1
        self.eye_distance_threshold_ratio = -1

        self.scale_factor = 0.3

        self.eye_left_slice: slice = slice(42, 48)
        self.eye_right_slice: slice = slice(36, 42)

        self.img_face: np.ndarray = np.ones((100, 100, 3), dtype=np.uint8)
        self.img_eye_left:  np.ndarray = np.ones((50, 50, 3), dtype=np.uint8)
        self.img_eye_right: np.ndarray = np.ones((50, 50, 3), dtype=np.uint8)

    def set_threshold(self, threshold: float):
        self.threshold = threshold

    def get_eye_left(self) -> str:
        return "closed" if self.left_closed else "open"

    def get_eye_right(self) -> str:
        return "closed" if self.right_closed else "open"

    def detect_eye_blinking_in_image(self, image: np.ndarray):
        # image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_sc = cv2.resize(gray, (int(image.shape[1] * self.scale_factor), int(image.shape[0] * self.scale_factor)), interpolation=cv2.INTER_LINEAR)
        # detect faces in the grayscale image
        rects = self.detector(gray_sc, 1)

        region_left = 0
        region_right = 0
        eye_distance = 1

        # loop over the face detections
        for i, rect in enumerate(rects):
            rect = scale_bbox(rect, 1/self.scale_factor)
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # draw the shapes into the face
            self.img_face = np.copy(image)
            eye_left  = shape[self.eye_left_slice]
            eye_right = shape[self.eye_right_slice]

            for (x, y) in eye_left:
                # left in blue color
                cv2.circle(self.img_face, (x, y), 1, (255, 0, 0), -1)
            for (x, y) in eye_right:
                # right eye in red color
                cv2.circle(self.img_face, (x, y), 1, (0, 0, 255), -1)

            # get the outer region of the plot

            eye_left_mean  = np.nanmean(eye_left, axis=0).astype(np.int32)
            eye_right_mean = np.nanmean(eye_right, axis=0).astype(np.int32)

            eye_left_width  = (np.nanmax(eye_left, axis=0)[0] - np.nanmin(eye_left, axis=0)[0]) // 2
            eye_right_width = (np.nanmax(eye_right, axis=0)[0] - np.nanmin(eye_right, axis=0)[0]) // 2

            bbox_eye_left = scale_bbox(
                bbox=dlib.rectangle(eye_left_mean[0] - eye_left_width, eye_left_mean[1] - eye_left_width, eye_left_mean[0] + eye_left_width, eye_left_mean[1] + eye_left_width,),
                scale=1,
                padding=-1
            )
            bbox_eye_right = scale_bbox(
                bbox=dlib.rectangle(eye_right_mean[0] - eye_right_width, eye_right_mean[1] - eye_right_width, eye_right_mean[0] + eye_right_width, eye_right_mean[1] + eye_right_width,),
                scale=1,
                padding=-1
            )

            self.img_eye_left  = self.img_face[bbox_eye_left.top():bbox_eye_left.bottom(), bbox_eye_left.left():bbox_eye_left.right()]
            self.img_eye_right = self.img_face[bbox_eye_right.top():bbox_eye_right.bottom(), bbox_eye_right.left():bbox_eye_right.right()]

            self.img_face = self.img_face[rect.top():rect.bottom(), rect.left():rect.right()]
            
            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # show the face number
            # cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            # for (x, y) in shape:
            #    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

            # calculate eye distance
            eye_distance = self.calculate_eye_distance(shape)
            #print(eye_distance)

            # calculate eye regions
            region_left, region_right = self.calculate_eye_regions(shape)
            #print(str(region_left) + '\t' + str(region_right))

            #print(self.threshold * eye_distance)
            #print(region_left)

            self.eye_distance_threshold_ratio = self.threshold * eye_distance
            
            # set closed to true if condition applies, else it will be false
            self.left_closed, self.right_closed = self.check_closing(region_left, region_right, eye_distance)

        self.left_eye_closing_norm_area = region_left / eye_distance
        self.right_eye_closing_norm_area = region_right / eye_distance

    def check_closing(self, region_left, region_right, eye_distance):
        return (region_left < (self.threshold * eye_distance), region_right < (self.threshold * eye_distance))


    def calculate_eye_regions(self, landmark_list):
        region_left = 0
        region_right = 0

        # eye triangles left
        t_l1_id = [42, 43, 47]
        t_l2_id = [43, 44, 47]
        t_l3_id = [44, 46, 47]
        t_l4_id = [44, 45, 46]
        # eye triangles right
        t_r1_id = [36, 38, 41]
        t_r2_id = [37, 38, 41]
        t_r3_id = [38, 40, 41]
        t_r4_id = [38, 39, 40]

        a_l1 = self.calculate_area_of_triangle(landmark_list[t_l1_id[0]], landmark_list[t_l1_id[1]],
                                          landmark_list[t_l1_id[2]])
        a_l2 = self.calculate_area_of_triangle(landmark_list[t_l2_id[0]], landmark_list[t_l2_id[1]],
                                          landmark_list[t_l2_id[2]])
        a_l3 = self.calculate_area_of_triangle(landmark_list[t_l3_id[0]], landmark_list[t_l3_id[1]],
                                          landmark_list[t_l3_id[2]])
        a_l4 = self.calculate_area_of_triangle(landmark_list[t_l4_id[0]], landmark_list[t_l4_id[1]],
                                          landmark_list[t_l4_id[2]])

        a_r1 = self.calculate_area_of_triangle(landmark_list[t_r1_id[0]], landmark_list[t_r1_id[1]],
                                          landmark_list[t_r1_id[2]])
        a_r2 = self.calculate_area_of_triangle(landmark_list[t_r2_id[0]], landmark_list[t_r2_id[1]],
                                          landmark_list[t_r2_id[2]])
        a_r3 = self.calculate_area_of_triangle(landmark_list[t_r3_id[0]], landmark_list[t_r3_id[1]],
                                          landmark_list[t_r3_id[2]])
        a_r4 = self.calculate_area_of_triangle(landmark_list[t_r4_id[0]], landmark_list[t_r4_id[1]],
                                          landmark_list[t_r4_id[2]])

        region_left = a_l1 + a_l2 + a_l3 + a_l4
        region_right = a_r1 + a_r2 + a_r3 + a_r4

        return region_left, region_right

    def calculate_area_of_triangle(self, p1, p2, p3):

        area = (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2.0

        return area

    def calculate_eye_distance(self, landmark_list):
        eye_distance = 0
        # left and right eye landmarks of both eyes
        r_id1 = 36
        r_id2 = 39
        l_id1 = 42
        l_id2 = 45
        # calculate the middle point of each eye
        eye_midpoint_r = (landmark_list[r_id1] + landmark_list[r_id2]) / 2.0
        eye_midpoint_l = (landmark_list[l_id1] + landmark_list[l_id2]) / 2.0
        # 2D Euclidean Distance
        eye_distance = np.sqrt((eye_midpoint_r[0] - eye_midpoint_l[0]) * (eye_midpoint_r[0] - eye_midpoint_l[0]) + (
                    eye_midpoint_r[1] - eye_midpoint_l[1]) * (eye_midpoint_r[1] - eye_midpoint_l[1]))

        return eye_distance

    def shape_to_np(self, shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype=dtype)
        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        # return the list of (x, y)-coordinates
        return coords
