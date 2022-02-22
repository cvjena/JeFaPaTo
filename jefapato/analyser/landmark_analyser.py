__all__ = ["LandmarkAnalyser"]

import collections
from typing import Any, List, OrderedDict, Type

import dlib
import numpy as np

from jefapato import extracting, features

from .abstract_analyser import hookspec
from .video_analyser import VideoAnalyser


class LandmarkAnalyser(VideoAnalyser):
    def __init__(self, features: List[Type[features.Feature]], **kwargs) -> None:
        super().__init__(extractor_c=extracting.LandmarkExtractor)

        kwargs["backend"] = kwargs.get("backend", "dlib")
        self.kwargs = kwargs

        self.results_file_header = "ear_score_left;ear_score_right;ear_valid\n"
        self.score_eye_l: List[float] = list()
        self.score_eye_r: List[float] = list()
        self.valid: List[bool] = list()

        self.feature_methods = collections.OrderedDict()
        self.feature_data = collections.OrderedDict()
        for feature in features:
            self.feature_methods[feature.__name__] = feature(**kwargs)
            self.feature_data[feature.__name__] = []

        self.current_frame = 0
        self.update_counter = -1
        self.update_skip = 20

    def set_frame_skip(self, value: int) -> None:
        self.update_skip = value

    def set_face_detect_skip(self, value: int) -> None:
        self.extractor.set_skip(value)

    def start(self) -> None:
        for m_name in self.feature_methods:
            self.feature_data[m_name].clear()

        self.analysis_setup()
        self.extractor.processingUpdated.connect(self.handle_update)
        self.analysis_start()

    def handle_update(
        self, data: np.ndarray, face_rect: Any, features: np.ndarray
    ) -> None:
        # here would be some drawing? and storing of the features we are interested in
        if self.kwargs["backend"] == "dlib":
            face_rect: dlib.rectangle = face_rect
        else:
            raise NotImplementedError("Only dlib backend is supported")

        self.pm.hook.updated(data=data, features=features)

        temp_data = collections.OrderedDict()

        for m_name, m_class in self.feature_methods.items():
            res = m_class.compute(features)
            self.feature_data[m_name].append(res)
            temp_data[m_name] = res

        self.pm.hook.updated_feature(features=temp_data)

    @hookspec
    def updated_feature(self, features: OrderedDict[str, Any]) -> None:
        pass

    # def __on_start(self):
    #     self.score_eye_l = list()
    #     self.score_eye_r = list()
    #     self.valid = list()

    #     if self.widget_graph is not None:
    #         self.widget_graph.start(self.get_fps())
    #         self.widget_graph.signal_graph_clicked.disconnect(self.__set_current_frame)

    # def __on_finished(self):
    #     if self.widget_graph is not None:
    #         self.widget_graph.finish(self.current_frame)
    #         self.widget_graph.signal_graph_clicked.connect(self.__set_current_frame)

    #     self.save_results()

    # def __on_update(self, frame: np.ndarray, frame_id: int):
    #     # currently we only check the first one
    #     # TODO make possible for more faces
    #     self.current_frame = frame_id
    #     self.append_classification()
    #     self.update_counter += 1

    #     if self.update_counter % self.update_skip != 0:
    #         return

    #     if self.widget_frame is not None:
    #         self.update_frame(frame)
    #     if self.widget_detail is not None:
    #         self.update_detail(frame)
    #     if self.widget_graph is not None:
    #         self.update_graph()

    # def append_classification(self) -> None:
    #     latest_value: EyeBlinkingResult = self.classes[-1][0]
    #     self.score_eye_l.append(latest_value.ear_left)
    #     self.score_eye_r.append(latest_value.ear_right)
    #     self.valid.append(latest_value.valid)

    # def update_graph(self) -> None:
    #     self.widget_graph.update(self.current_frame)

    #     if self.curve_l is not None:
    #         self.curve_l.setData(self.score_eye_l)
    #     if self.curve_r is not None:
    #         self.curve_r.setData(self.score_eye_r)

    # def update_frame(self, frame: np.ndarray) -> None:
    #     img = np.copy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #     self.widget_frame.frame.set_image(img)

    def __set_current_frame(self, frame: float) -> None:
        pass

    #     # this function get only called if a graph obj exists!
    #     # clip the frame into the maximum valid range
    #     # cast to int, just to be sure
    #     value = int(
    #         max(
    #             0,
    #             min(int(frame), self.data_amount - 1),
    #         )
    #     )
    #     self.widget_graph.set_x_ruler(value)
    #     self.current_frame = value

    #     if not self.resource_is_loaded():
    #         return

    #     self.set_next_item_by_id(self.current_frame)
    #     (grabbed, frame) = self.get_next_item()

    #     # TODO do something better if nothing gets grabbed
    #     if not grabbed:
    #         return

    #     if self.widget_frame is not None:
    #         self.update_frame(frame)
    #     if self.widget_detail is not None:
    #         self.update_detail(frame)

    # def update_detail(self, frame: np.ndarray) -> None:
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     try:
    #         rect, shape = self.features[self.current_frame][0]
    #     except IndexError:
    #         rect, shape = None, None

    #     try:
    #         self.widget_detail.set_labels(
    #             str(round(self.score_eye_l[self.current_frame], 4)),
    #             str(round(self.score_eye_r[self.current_frame], 4)),
    #         )
    #     except IndexError:
    #         pass
    #     self.widget_detail.set_frame(frame, rect, shape)

    # def save_results(self):
    #     # TODO make this saving way more generic
    #     resource_path = self.get_resource_path()
    #     ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #     result_path = resource_path.parent / (resource_path.stem + f"_{ts}.csv")

    #     with open(result_path, "w") as f:
    #         f.write(self.results_file_header)
    #         for i in range(len(self.score_eye_r)):
    #             # fancy String literal concatenation
    #             line = (
    #                 f"{self.score_eye_l[i]};"
    #                 f"{self.score_eye_r[i]};"
    #                 f"{self.valid[i]}"
    #                 f"\n"
    #             )
    #             f.write(line)

    def call_after_resource_load(self) -> None:
        self.reset()
        self.__set_current_frame(self.current_frame)
