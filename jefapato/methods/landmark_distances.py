import numpy as np


class LandmarkDistance3D:
    def __init__(self):
        super().__init__()

        self.landmark_id1_left = 0
        self.landmark_id2_left = 0
        self.landmark_id1_right = 0
        self.landmark_id2_right = 0

        self.distance_left = 0
        self.distance_right = 0

        self.threshold = 0

        self.landmark_coordinates = []
        self.landmark_file_path = None

    def set_threshold(self, threshold):
        self.threshold = threshold

    def set_landmark_file_path(self, landmark_file_path):
        self.landmark_file_path = landmark_file_path

    def set_landmark_ids(
        self, lm_idx1_left, lm_idx2_left, lm_idx1_right, lm_idx2_right
    ):
        self.landmark_id1_left = lm_idx1_left - 1
        self.landmark_id2_left = lm_idx2_left - 1
        self.landmark_id1_right = lm_idx1_right - 1
        self.landmark_id2_right = lm_idx2_right - 1

    def read_landmark_file(self, landmark_file_path):
        self.landmark_file_path = landmark_file_path
        with open(self.landmark_file_path) as f:
            content = f.readlines()
            # you may also want to remove whitespace characters like `\n` at the
            # end of each line
            content = [x.strip() for x in content]

            self.landmark_coordinates = []

            for frame_idx, frame in enumerate(content):
                frame_values = frame.split(" ")
                frame_values.pop(0)
                frame_values = [x for x in frame_values if x]
                landmarks_in_frame = []

                for value_idx, value in enumerate(frame_values):
                    if value_idx % 3 == 0:
                        lm = [
                            float(frame_values[value_idx]),
                            float(frame_values[value_idx + 1]),
                            float(frame_values[value_idx + 2]),
                        ]
                        landmarks_in_frame.append(lm)
                self.landmark_coordinates.append(landmarks_in_frame)

        return self.landmark_coordinates

    def get_distances(self):
        self.distances_left = []
        self.distances_right = []
        for frame_idx, frame in enumerate(self.landmark_coordinates):
            self.distance_left = np.linalg.norm(
                np.asarray(frame[self.landmark_id1_left])
                - np.asarray(frame[self.landmark_id2_left])
            )
            self.distances_left.append(self.distance_left)
            self.distance_right = np.linalg.norm(
                np.asarray(frame[self.landmark_id1_right])
                - np.asarray(frame[self.landmark_id2_right])
            )
            self.distances_right.append(self.distance_right)
        return self.distances_left, self.distances_right

    def get_distances_from_frame(self, frame_idx):
        if len(self.distances_left) == 0:
            self.get_distances()
        self.distance_left = self.distances_left[frame_idx]
        self.distance_right = self.distances_right[frame_idx]
        return self.distance_left, self.distance_right

    def get_distance(self, lm1, lm2):
        distance = np.linalg.norm(lm1 - lm2)
        return distance


class LandmarkDistance2D:
    def __init__(self):
        super().__init__()

    def get_distance(self, lm1, lm2):
        distance = np.linalg.norm(lm1 - lm2)
        return distance
