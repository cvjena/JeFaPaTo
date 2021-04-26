import os
import sys
import numpy as np
import dlib
import cv2
import getopt
import argparse
import glob
import imutils
from imutils import face_utils


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

    def set_threshold(self, threshold):
        self.threshold = threshold

    def detect_eye_blinking_in_image(self, image):
        # image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale image
        rects = self.detector(gray, 1)

        # loop over the face detections
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
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

            if region_left < self.threshold * eye_distance:
                self.left_closed = True
            else:
                self.left_closed = False
            if region_right < self.threshold * eye_distance:
                self.right_closed = True
            else:
                self.right_closed = False

        self.left_eye_closing_norm_area = region_left / eye_distance
        self.right_eye_closing_norm_area = region_right / eye_distance

        return self.left_closed, self.right_closed, self.left_eye_closing_norm_area, self.right_eye_closing_norm_area

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
