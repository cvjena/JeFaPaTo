# __all__ = ["FrameWidget", "FrameViewBox", "EyeDetailWidget", "GraphWidget"]

# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional

# import cv2
# import dlib
# import numpy as np
# import pyqtgraph as pg
# from PyQt5.QtCore import pyqtSignal

# IMAGE_SETTINGS = {
#     "invertY": True,
#     "lockAspect": True,
#     "enableMouse": False,
# }


# @dataclass
# class BoundingBox:
#     l: int = 0
#     r: int = -1
#     t: int = 0
#     b: int = -1


# class FrameWidget(pg.GraphicsLayoutWidget):
#     def __init__(self, parent=None, show=False, size=None, title=None, **kargs):
#         super().__init__(parent=parent, show=show, size=size, title=title, **kargs)

#         self.vb = FrameViewBox(**IMAGE_SETTINGS)
#         self.addItem(self.vb)
#         self.frame = self.vb


# class EyeDetailWidget(pg.GraphicsLayoutWidget):
#     def __init__(self, parent=None, show=False, size=None, title=None, **kargs):
#         super().__init__(parent=parent, show=show, size=size, title=title, **kargs)

#         self.layout_l = pg.GraphicsLayout()
#         self.layout_r = pg.GraphicsLayout()

#         self.frame_d = FrameViewBox(**IMAGE_SETTINGS)
#         self.frame_l = FrameViewBox(**IMAGE_SETTINGS)
#         self.frame_r = FrameViewBox(**IMAGE_SETTINGS)

#         # label
#         self.label_l: pg.LabelItem = pg.LabelItem("open")
#         self.label_r: pg.LabelItem = pg.LabelItem("open")

#         self.addItem(self.frame_d, row=0, col=0, rowspan=2, colspan=2)
#         self.addItem(self.layout_l, row=2, col=1)
#         self.addItem(self.layout_r, row=2, col=0)

#         # detail right eye
#         self.layout_r.addLabel(text="Right eye", row=0, col=0)
#         self.layout_r.addItem(self.frame_r, row=1, col=0)
#         self.layout_r.addItem(self.label_r, row=2, col=0)

#         # detail left eye
#         self.layout_l.addLabel(text="Left eye", row=0, col=0)
#         self.layout_l.addItem(self.frame_l, row=1, col=0)
#         self.layout_l.addItem(self.label_l, row=2, col=0)

#     def set_labels(self, left: str, right: str) -> None:
#         self.label_l.setText(left)
#         self.label_r.setText(right)

#     def set_frame(
#         self,
#         frame: np.ndarray,
#         rect: dlib.rectangle,
#         shape: np.ndarray,
#         eye_padding: float = 1.5,
#     ) -> None:
#         self.frame_d.set_image(np.zeros((20, 20)))
#         self.frame_l.set_image(np.zeros((20, 20)))
#         self.frame_r.set_image(np.zeros((20, 20)))

#         if rect is not None or shape is not None:
#             eye_l = shape[slice(42, 48)]
#             eye_r = shape[slice(36, 42)]

#             eye_l_m = np.nanmean(eye_l, axis=0).astype(np.int32)
#             eye_r_m = np.nanmean(eye_r, axis=0).astype(np.int32)

#             eye_l_w = (np.nanmax(eye_l, axis=0)[0] - np.nanmin(eye_l, axis=0)[0]) // 2
#             eye_r_w = (np.nanmax(eye_r, axis=0)[0] - np.nanmin(eye_r, axis=0)[0]) // 2
#             eye_l_w = int(eye_padding * eye_l_w)
#             eye_r_w = int(eye_padding * eye_r_w)

#             crop = eye_l_w if eye_l_w > eye_r_w else eye_r_w

#             bbox_l = BoundingBox(
#                 l=eye_l_m[0] - crop,
#                 r=eye_l_m[0] + crop,
#                 t=eye_l_m[1] - crop,
#                 b=eye_l_m[1] + crop,
#             )
#             bbox_r = BoundingBox(
#                 l=eye_r_m[0] - crop,
#                 r=eye_r_m[0] + crop,
#                 t=eye_r_m[1] - crop,
#                 b=eye_r_m[1] + crop,
#             )

#             bbox_f = BoundingBox(
#                 l=bbox_r.l,
#                 r=bbox_l.r,
#                 t=min(bbox_l.t, bbox_r.t),
#                 b=max(bbox_l.b, bbox_r.b),
#             )

#             self.draw_points(frame, eye_l, (0, 0, 255))
#             self.draw_points(frame, eye_r, (255, 0, 0))

#             self.draw_ratio(frame, eye_l, (0, 0, 255))
#             self.draw_ratio(frame, eye_r, (255, 0, 0))

#             self.frame_d.set_image(frame, bbox_f)
#             self.frame_l.set_image(frame, bbox_l)
#             self.frame_r.set_image(frame, bbox_r)

#     def draw_points(self, frame, shape, color):
#         for (x, y) in shape:
#             cv2.circle(frame, (x, y), 1, color, -1)

#     def draw_ratio(self, frame, shape, color):
#         # horizontal line
#         cv2.line(
#             frame, (shape[0][0], shape[0][1]), (shape[3][0], shape[3][1]), color, 1
#         )

#         # vertical line
#         t = (shape[1] + shape[2]) // 2
#         b = (shape[4] + shape[5]) // 2
#         cv2.line(frame, (t[0], t[1]), (b[0], b[1]), color, 1)
