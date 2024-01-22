import time
from pathlib import Path

import cv2
import pytest

from jefapato.facial_features import mediapipe_landmark_extractor
from jefapato.facial_features import video_data_loader 

from xvfbwrapper import Xvfb # for headless rendering
vdisplay = Xvfb(width=512, height=512)
vdisplay.start()

class ExtractorHook:
    def __init__(self) -> None:
        self.triggered_handle_update = False
        self.triggered_handle_finished = False
        self.triggered_update_progress = False
        self.triggered_handle_pause = False
        self.triggered_handle_resume = False
    
    @mediapipe_landmark_extractor.MediapipeLandmarkExtractor.hookimpl
    def handle_update(self, item) -> None:
        self.triggered_handle_update = True
    
    @mediapipe_landmark_extractor.MediapipeLandmarkExtractor.hookimpl
    def handle_finished(self) -> None:
        self.triggered_handle_finished = True
    
    @mediapipe_landmark_extractor.MediapipeLandmarkExtractor.hookimpl
    def update_progress(self, perc: int) -> None:
        self.triggered_update_progress = True

    @mediapipe_landmark_extractor.MediapipeLandmarkExtractor.hookimpl
    def handle_pause(self) -> None:
        self.triggered_handle_pause = True
    
    @mediapipe_landmark_extractor.MediapipeLandmarkExtractor.hookimpl
    def handle_resume(self) -> None:
        self.triggered_handle_resume = True

def get_test_video_path() -> Path:
    return Path(__file__).parent / "files/test_10sec.mp4"


def test_extractor_without_items():
    video = cv2.VideoCapture(str(get_test_video_path()))
    loader = video_data_loader.VideoDataLoader(video.read, data_amount=200, queue_maxsize=100)

    extractor = mediapipe_landmark_extractor.MediapipeLandmarkExtractor(
        data_queue=loader.data_queue,
        data_amount=loader.data_amount,
    )
    
    hook = ExtractorHook()
    extractor.register(hook)
    with pytest.raises(ValueError):
        extractor.register(hook)
    
    extractor.start()
    extractor.join()
    
    assert not hook.triggered_handle_pause
    assert not hook.triggered_handle_resume
    assert not hook.triggered_handle_update # no items to process so no update should be triggered
    assert hook.triggered_handle_finished
    assert hook.triggered_update_progress
    
def test_extractor_pause_resume():
    video = cv2.VideoCapture(str(get_test_video_path()))
    loader = video_data_loader.VideoDataLoader(video.read, data_amount=200, queue_maxsize=100)

    extractor = mediapipe_landmark_extractor.MediapipeLandmarkExtractor(
        data_queue=loader.data_queue,
        data_amount=loader.data_amount,
    )
    
    hook = ExtractorHook()
    extractor.register(hook)
    with pytest.raises(ValueError):
        extractor.register(hook)
    
    loader.start()
    extractor.start()
    
    assert extractor.isRunning()
    extractor.pause()
    time.sleep(1.0)
    assert not extractor.isRunning()
    extractor.resume()
    assert extractor.isRunning()
    
    extractor.join()
    
    assert hook.triggered_handle_update
    assert hook.triggered_handle_finished
    assert hook.triggered_update_progress
    assert hook.triggered_handle_pause
    assert hook.triggered_handle_resume
    
    
def test_extractor_toggle_pause():
    video = cv2.VideoCapture(str(get_test_video_path()))
    loader = video_data_loader.VideoDataLoader(video.read, data_amount=200, queue_maxsize=100)

    extractor = mediapipe_landmark_extractor.MediapipeLandmarkExtractor(
        data_queue=loader.data_queue,
        data_amount=loader.data_amount,
        bbox_slice=(166, 662, 694, 1190),
    )
    
    hook = ExtractorHook()
    extractor.register(hook)
    with pytest.raises(ValueError):
        extractor.register(hook)
    
    loader.start()
    extractor.start()
    time.sleep(1.0)
    extractor.toggle_pause()
    assert not extractor.isRunning()
    extractor.toggle_pause()
    assert extractor.isRunning()
    
    extractor.join()
    
    assert hook.triggered_handle_update
    assert hook.triggered_handle_finished
    assert hook.triggered_update_progress
    assert hook.triggered_handle_pause
    assert hook.triggered_handle_resume


def test_extractor_force_stop():
    video = cv2.VideoCapture(str(get_test_video_path()))
    loader = video_data_loader.VideoDataLoader(video.read, data_amount=200, queue_maxsize=100)

    extractor = mediapipe_landmark_extractor.MediapipeLandmarkExtractor(
        data_queue=loader.data_queue,
        data_amount=loader.data_amount,
        bbox_slice=(166, 662, 694, 1190),
    )
    
    hook = ExtractorHook()
    extractor.register(hook)
    with pytest.raises(ValueError):
        extractor.register(hook)
    
    loader.start()
    extractor.start()
    
    time.sleep(1)
    extractor.stop()
    extractor.join()
    
    assert hook.triggered_handle_update
    assert hook.triggered_handle_finished
    assert hook.triggered_update_progress
    assert not hook.triggered_handle_pause
    assert not hook.triggered_handle_resume
    
    assert extractor.data_amount >= extractor.processed
