from collections import OrderedDict
from pathlib import Path
from threading import Thread

import numpy as np
import pandas as pd
import pytest

from jefapato import facial_features  

from xvfbwrapper import Xvfb # for headless rendering
vdisplay = Xvfb(width=512, height=512)
vdisplay.start()

def get_test_video_path() -> Path:
    return Path(__file__).parent / "files/test_10sec.mp4"

def get_test_vide_path_corrupt() -> Path:
    return Path(__file__).parent / "files/test_10sec_corrupt.mp4"

def get_test_fail_file() -> Path:
    return Path(__file__).parent / "files/test_file.mp3"


def get_format_files() -> list[Path]:
    return list((Path(__file__).parent / "files/formats/").iterdir())

def get_temp_folder() -> Path:
    return Path(__file__).parent / "temp_results"

@pytest.fixture(autouse=False, scope="function")
def clean_folder():
    yield
    folder = get_temp_folder()
    if folder.exists():
        for file in folder.iterdir():
            file.unlink()
        folder.rmdir()

class HookTest:
    def __init__(self) -> None:
        self.hook_started = False
        self.hook_paused = False
        self.hook_resumed = False
        self.hook_processed_percentage = False
        self.hook_updated_feature = False
        self.hook_updated_display = False

    @facial_features.FaceAnalyzer.hookimpl
    def started(self):
        self.hook_started = True

    @facial_features.FaceAnalyzer.hookimpl
    def paused(self):
        self.hook_paused = True

    @facial_features.FaceAnalyzer.hookimpl
    def resumed(self):
        self.hook_resumed = True

    @facial_features.FaceAnalyzer.hookimpl
    def processed_percentage(self, percentage: int):
        self.hook_processed_percentage = True

    @facial_features.FaceAnalyzer.hookimpl
    def finished(self):
        self.hook_finished = True

    @facial_features.FaceAnalyzer.hookimpl
    def updated_feature(self, feature_data: OrderedDict):
        self.hook_updated_feature = True

    @facial_features.FaceAnalyzer.hookimpl
    def updated_display(self, image: np.ndarray):
        self.hook_updated_display = True


def test_analyzer_init():
    analyzer = facial_features.FaceAnalyzer()

    assert analyzer.max_ram_size == 4 << 28
    assert analyzer.feature_classes == {}
    assert analyzer.feature_data == {}
    assert analyzer.resource_interface is None
    assert analyzer.video_resource is None
    assert analyzer.data_amount == 0
    assert not hasattr(analyzer, "loader") and not hasattr(analyzer, "extractor")


def test_analyzer_resource():
    analyzer = facial_features.FaceAnalyzer()

    with pytest.raises(ValueError):
        analyzer.prepare_video_resource("")

    with pytest.raises(ValueError):
        analyzer.prepare_video_resource(None)

    with pytest.raises(ValueError):
        analyzer.prepare_video_resource(1)

    with pytest.raises(ValueError):
        analyzer.prepare_video_resource(1.0)

    with pytest.raises(ValueError):
        analyzer.prepare_video_resource(get_test_video_path().as_posix())

    with pytest.raises(ValueError):
        analyzer.prepare_video_resource(get_test_video_path().as_uri())

    with pytest.raises(ValueError):
        analyzer.prepare_video_resource(get_test_video_path().parent)

    # test with non-existing file
    with pytest.raises(FileNotFoundError):
        analyzer.prepare_video_resource(Path("non-existing-file.mp4"))

    with pytest.raises(ValueError):
        analyzer.prepare_video_resource(get_test_fail_file())

    success, img = analyzer.prepare_video_resource(get_test_video_path())

    assert success
    assert isinstance(img, np.ndarray)

    # test the data_amount

    assert analyzer.data_amount == 270


def test_analyzer_resource_formats():
    # should not raise any errors
    analyzer = facial_features.FaceAnalyzer()
    for file in get_format_files():
        analyzer.prepare_video_resource(file)


def test_analyzer_analysis_setup():
    analyzer = facial_features.FaceAnalyzer()

    with pytest.raises(ValueError):
        analyzer.analysis_setup()

    analyzer.prepare_video_resource(get_test_video_path())

    # check the output
    assert analyzer.analysis_setup()

    assert analyzer.loader is not None
    assert isinstance(analyzer.loader, Thread)
    assert analyzer.extractor is not None
    assert isinstance(analyzer.extractor, Thread)


def test_analyzer_analysis_start():
    analyzer = facial_features.FaceAnalyzer()

    with pytest.raises(RuntimeError):
        analyzer.analysis_start()

    analyzer.prepare_video_resource(get_test_video_path())
    analyzer.analysis_setup()
    analyzer.analysis_start()

    analyzer.stop()


def test_analyzer_analysis_start_with_hook():
    analyzer = facial_features.FaceAnalyzer()
    hook_test = HookTest()
    analyzer.register_hooks(hook_test)

    analyzer.prepare_video_resource(get_test_video_path())
    analyzer.analysis_setup()
    analyzer.analysis_start()

    analyzer.stop()
    assert hook_test.hook_started


def test_analyzer_stop_without_start():
    analyzer = facial_features.FaceAnalyzer()
    analyzer.stop()


def test_analyzer_register_hook():
    analyzer = facial_features.FaceAnalyzer()

    hook_1 = HookTest()
    hook_2 = HookTest()

    analyzer.register_hooks(hook_1)
    assert len(analyzer.pm.get_plugins()) == 1
    analyzer.register_hooks(hook_2)
    assert len(analyzer.pm.get_plugins()) == 2


def test_analyzer_register_hook_twice():
    analyzer = facial_features.FaceAnalyzer()

    hook = HookTest()

    analyzer.register_hooks(hook)
    analyzer.register_hooks(hook)


def test_analyzer_set_features():
    analyzer = facial_features.FaceAnalyzer()

    with pytest.raises(ValueError):
        analyzer.set_features(None)

    with pytest.raises(ValueError):
        analyzer.set_features(1)

    with pytest.raises(ValueError):
        analyzer.set_features("test")

    with pytest.raises(ValueError):
        analyzer.set_features(["test"])

    with pytest.raises(ValueError):
        analyzer.set_features([1])

    with pytest.raises(ValueError):
        analyzer.set_features([None])

    with pytest.raises(ValueError):
        analyzer.set_features([1, None])

    with pytest.raises(ValueError):
        analyzer.set_features([1, "test"])

    with pytest.raises(ValueError):
        analyzer.set_features([1, 1])

    with pytest.raises(ValueError):
        analyzer.set_features([HookTest])

    analyzer.set_features([facial_features.features.EAR2D6, facial_features.features.EAR3D6])
    assert len(analyzer.feature_classes) == 2

    analyzer.set_features([facial_features.features.EAR2D6, facial_features.features.EAR2D6])
    assert len(analyzer.feature_classes) == 1

    for feature in analyzer.feature_classes:
        assert len(analyzer.feature_data[feature]) == 0


def test_analyzer_set_features_after_start():
    analyzer = facial_features.FaceAnalyzer()

    analyzer.prepare_video_resource(get_test_video_path())
    analyzer.analysis_setup()
    analyzer.analysis_start()

    with pytest.raises(RuntimeError):
        analyzer.set_features([facial_features.features.EAR2D6, facial_features.features.EAR3D6])
    analyzer.stop()


def test_analyzer_set_features_after_stop():
    analyzer = facial_features.FaceAnalyzer()

    analyzer.prepare_video_resource(get_test_video_path())
    analyzer.analysis_setup()
    analyzer.analysis_start()
    analyzer.stop()
    analyzer.set_features([facial_features.features.EAR2D6, facial_features.features.EAR3D6])


def test_analyzer_toggle_pause():
    analyzer = facial_features.FaceAnalyzer()
    hook = HookTest()
    analyzer.register_hooks(hook)

    analyzer.prepare_video_resource(get_test_video_path())
    analyzer.analysis_setup()
    analyzer.set_features([facial_features.features.EAR2D6, facial_features.features.EAR3D6])
    analyzer.analysis_start()

    assert not analyzer.extractor.paused
    analyzer.toggle_pause()
    assert analyzer.extractor.paused
    analyzer.toggle_pause()
    assert not analyzer.extractor.paused

    analyzer.stop()

    assert hook.hook_paused


def test_analyzer_toggle_pause_without_setup():
    analyzer = facial_features.FaceAnalyzer()

    analyzer.prepare_video_resource(get_test_video_path())
    analyzer.set_features([facial_features.features.EAR2D6, facial_features.features.EAR3D6])
    with pytest.raises(RuntimeError):
        analyzer.toggle_pause()


def test_analyzer_toggle_pause_without_start():
    analyzer = facial_features.FaceAnalyzer()

    analyzer.prepare_video_resource(get_test_video_path())
    analyzer.analysis_setup()
    analyzer.set_features([facial_features.features.EAR2D6, facial_features.features.EAR3D6])

    analyzer.toggle_pause()


def test_analyzer_toggle_pause_after_stop():
    analyzer = facial_features.FaceAnalyzer()

    analyzer.prepare_video_resource(get_test_video_path())
    analyzer.analysis_setup()
    analyzer.set_features([facial_features.features.EAR2D6, facial_features.features.EAR3D6])
    analyzer.analysis_start()
    analyzer.stop()

    analyzer.toggle_pause()


def test_analyzer_start():
    analyzer = facial_features.FaceAnalyzer()

    analyzer.prepare_video_resource(get_test_video_path())
    analyzer.set_features([facial_features.features.EAR2D6, facial_features.features.EAR3D6])
    analyzer.clean_start()  # combines analysis_setup() and analysis_start() in one method
    analyzer.stop()


def test_analyzer_throughput():
    analyzer = facial_features.FaceAnalyzer()

    analyzer.prepare_video_resource(get_test_video_path())
    analyzer.analysis_setup()
    analyzer.set_features([facial_features.features.EAR2D6, facial_features.features.EAR3D6])
    analyzer.analysis_start()

    data_input, data_procssing = analyzer.get_throughput()

    assert isinstance(data_input, int) and data_input >= 0
    assert isinstance(data_procssing, int) and data_procssing >= 0
    analyzer.stop()


def test_analyzer_throughput_without_setup():
    analyzer = facial_features.FaceAnalyzer()

    analyzer.prepare_video_resource(get_test_video_path())
    analyzer.set_features([facial_features.features.EAR2D6, facial_features.features.EAR3D6])

    with pytest.raises(RuntimeError):
        analyzer.get_throughput()


def test_analyzer_throughput_without_start():
    analyzer = facial_features.FaceAnalyzer()

    analyzer.prepare_video_resource(get_test_video_path())
    analyzer.analysis_setup()
    analyzer.set_features([facial_features.features.EAR2D6, facial_features.features.EAR3D6])

    data_input, data_procssing = analyzer.get_throughput()

    assert isinstance(data_input, int) and data_input == 0
    assert isinstance(data_procssing, int) and data_procssing == 0


def test_analyzer_get_item_size():
    analyzer = facial_features.FaceAnalyzer()
    analyzer.prepare_video_resource(get_test_video_path())

    w, h, c = analyzer.get_item_size()

    assert isinstance(w, int) and w > 0 and w == 1920
    assert isinstance(h, int) and h > 0 and h == 1080
    assert isinstance(c, int) and c > 0 and c == 3

    b = analyzer.get_item_size(in_bytes=True)
    assert isinstance(b, int) and b > 0 and b == 1920 * 1080 * 3


def test_analyzer_get_fps():
    analyzer = facial_features.FaceAnalyzer()
    analyzer.prepare_video_resource(get_test_video_path())

    fps = analyzer.get_fps()

    assert isinstance(fps, float) and fps > 0 and fps == 30.0


def test_analyzer_full_run():
    analyzer = facial_features.FaceAnalyzer()
    hook = HookTest()
    analyzer.register_hooks(hook)

    analyzer.prepare_video_resource(get_test_video_path())
    analyzer.analysis_setup()
    analyzer.set_features([facial_features.features.EAR2D6, facial_features.features.EAR3D6, facial_features.features.BS_EyeLookDownLeft])
    analyzer.analysis_start()

    analyzer.extractor.join()

@pytest.mark.usefixtures("clean_folder")
def test_analyzer_save_after_full_run():
    analyzer = facial_features.FaceAnalyzer()
    hook = HookTest()
    analyzer.register_hooks(hook)
    
    result_folder = get_temp_folder()

    analyzer.prepare_video_resource(get_test_video_path())
    analyzer.analysis_setup()
    analyzer.set_features([facial_features.features.EAR2D6, facial_features.features.EAR3D6, facial_features.features.BS_EyeLookDownLeft])
    analyzer.analysis_start()

    analyzer.extractor.join()
    with pytest.raises(ValueError):
        analyzer.save_results("")

    with pytest.raises(ValueError):
        analyzer.save_results(None)

    with pytest.raises(ValueError):
        analyzer.save_results(1)

    with pytest.raises(ValueError):
        analyzer.save_results(1.0)

    with pytest.raises(ValueError):
        analyzer.save_results(result_folder.as_posix())

    assert analyzer.save_results(result_folder)

    # check the output
    files = list(result_folder.iterdir())
    assert len(files) == 1
    assert files[0].suffix == ".csv"
    assert files[0].stem.startswith("test_10sec")

    # import the file with pandas
    df = pd.read_csv(files[0])
    # check the columns
    cols = list(df.columns)
    assert "frame" in cols
    assert "EAR2D6_l" in cols
    assert "EAR2D6_r" in cols
    assert "EAR2D6_valid" in cols
    assert "EAR3D6_l" in cols
    assert "EAR3D6_r" in cols
    assert "EAR3D6_valid" in cols
    assert "BS_EyeLookDownLeft" in cols
    assert "BS_Valid" in cols
    # check the amount of rows
    assert len(df) == (270 - 1)  # 270 frames - 1 frame for the last one

@pytest.mark.usefixtures("clean_folder")
def test_analyzer_save_after_full_run_without_features():
    # clean up, if there is any due to previous test
    result_folder = get_temp_folder()

    analyzer = facial_features.FaceAnalyzer()
    hook = HookTest()
    analyzer.register_hooks(hook)

    analyzer.prepare_video_resource(get_test_video_path())
    analyzer.analysis_setup()
    analyzer.analysis_start()
    analyzer.extractor.join()

    assert analyzer.save_results(result_folder)

    # check the output
    files = list(result_folder.iterdir())
    assert len(files) == 1
    assert files[0].suffix == ".csv"
    assert files[0].stem.startswith("test_10sec")

@pytest.mark.usefixtures("clean_folder")
def test_analyzer_full_run_with_bbox():
    analyzer = facial_features.FaceAnalyzer()
    hook = HookTest()
    analyzer.register_hooks(hook)

    analyzer.prepare_video_resource(get_test_video_path())
    analyzer.analysis_setup(bbox_slice=(166, 662, 694, 1190))
    analyzer.set_features([facial_features.features.EAR2D6, facial_features.features.EAR3D6, facial_features.features.BS_EyeLookDownLeft])
    analyzer.analysis_start()
    analyzer.extractor.join()

    result_folder = get_temp_folder()
    analyzer.save_results(result_folder)
    # check the output
    files = list(result_folder.iterdir())
    assert len(files) == 1
    assert files[0].suffix == ".csv"
    assert files[0].stem.startswith("test_10sec")
    # import the file with pandas
    df = pd.read_csv(files[0])
    # check the columns
    cols = list(df.columns)
    assert "frame" in cols
    assert "EAR2D6_l" in cols
    assert "EAR2D6_r" in cols
    assert "EAR2D6_valid" in cols
    assert "EAR3D6_l" in cols
    assert "EAR3D6_r" in cols
    assert "EAR3D6_valid" in cols
    assert "BS_EyeLookDownLeft" in cols
    assert "BS_Valid" in cols
    # check the amount of rows
    assert len(df) == (270 - 1)  # 270 frames - 1 frame for the last one
    
    # all EAR values should be 1.0
    assert np.all(df["EAR2D6_l"].values != 1.0)
    assert np.all(df["EAR2D6_r"].values != 1.0)
    assert np.all(df["EAR3D6_l"].values != 1.0)
    assert np.all(df["EAR3D6_r"].values != 1.0)
    assert np.all(df["EAR2D6_valid"].values)
    assert np.all(df["EAR3D6_valid"].values)

@pytest.mark.usefixtures("clean_folder")
def test_analyzer_full_run_with_wrong_bbox():
    analyzer = facial_features.FaceAnalyzer()
    hook = HookTest()
    analyzer.register_hooks(hook)

    analyzer.prepare_video_resource(get_test_video_path())
    analyzer.analysis_setup(bbox_slice=(55, 551, 50, 546))
    analyzer.set_features([facial_features.features.EAR2D6, facial_features.features.EAR3D6, facial_features.features.BS_EyeLookDownLeft])
    analyzer.analysis_start()
    analyzer.extractor.join()

    result_folder = get_temp_folder()
    analyzer.save_results(result_folder)
    # check the output
    files = list(result_folder.iterdir())
    assert len(files) == 1
    assert files[0].suffix == ".csv"
    assert files[0].stem.startswith("test_10sec")
    # import the file with pandas
    df = pd.read_csv(files[0])
    # check the columns
    cols = list(df.columns)
    assert "frame" in cols
    assert "EAR2D6_l" in cols
    assert "EAR2D6_r" in cols
    assert "EAR2D6_valid" in cols
    assert "EAR3D6_l" in cols
    assert "EAR3D6_r" in cols
    assert "EAR3D6_valid" in cols
    assert "BS_EyeLookDownLeft" in cols
    assert "BS_Valid" in cols
    # check the amount of rows
    assert len(df) == (270 - 1)  # 270 frames - 1 frame for the last one
    
    # all EAR values should be 1.0
    assert np.allclose(df["EAR2D6_l"].values, 1.0)
    assert np.allclose(df["EAR2D6_r"].values, 1.0)
    assert np.allclose(df["EAR3D6_l"].values, 1.0)
    assert np.allclose(df["EAR3D6_r"].values, 1.0)
    assert not np.all(df["EAR2D6_valid"].values)
    assert not np.all(df["EAR3D6_valid"].values)

def test_analyzer_save_before_and_during_full_run():
    analyzer = facial_features.FaceAnalyzer()
    result_folder = get_temp_folder()

    with pytest.raises(RuntimeError):  # before start
        analyzer.save_results(result_folder)

    analyzer.prepare_video_resource(get_test_video_path())
    analyzer.analysis_setup()

    analyzer.set_features([facial_features.features.EAR2D6, facial_features.features.EAR3D6, facial_features.features.BS_EyeLookDownLeft])
    analyzer.analysis_start()

    with pytest.raises(RuntimeError):
        analyzer.save_results(result_folder)

    analyzer.extractor.join()

def test_analyzer_corrupt_video():
    analyzer = facial_features.FaceAnalyzer()
    analyzer.prepare_video_resource(get_test_vide_path_corrupt())
    analyzer.analysis_setup()
    analyzer.set_features([facial_features.features.EAR2D6])
    analyzer.analysis_start()

    analyzer.extractor.join()


@pytest.mark.usefixtures("clean_folder")
def test_analyzer_full_run_most_features():
    analyzer = facial_features.FaceAnalyzer()
    hook = HookTest()
    analyzer.register_hooks(hook)

    analyzer.prepare_video_resource(get_test_video_path())
    analyzer.analysis_setup(bbox_slice=(166, 662, 694, 1190))    
    analyzer.set_features([
        facial_features.features.EAR2D6, 
        facial_features.features.EAR3D6, 
        facial_features.features.BS_EyeLookDownLeft,
        facial_features.features.Landmarks68,
        facial_features.features.Landmarks478,
    ])

    analyzer.analysis_start()
    analyzer.extractor.join()

    result_folder = get_temp_folder()
    analyzer.save_results(result_folder)
    # check the output
    files = list(result_folder.iterdir())
    assert len(files) == 1
    assert files[0].suffix == ".csv"
    assert files[0].stem.startswith("test_10sec")
    
    # import the file with pandas
    df = pd.read_csv(files[0])
    # check the columns
    cols = list(df.columns)
    assert len(cols) == 1 + 3 + 3 + 2 + (68+1) + (478+1) 
