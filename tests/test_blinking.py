import math
from pathlib import Path
import numpy as np
import pandas as pd

import pytest

from jefapato import blinking

from xvfbwrapper import Xvfb # for headless rendering
vdisplay = Xvfb(width=512, height=512)
vdisplay.start()

def get_file_short():
    return Path(__file__).parent / "files" / "blinking" / "blinking_short.csv"

def get_file_full():
    return Path(__file__).parent / "files" / "blinking" / "blinking_full.csv"

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

def test_peaks_with_valid_input():
    ear_l, ear_r = blinking.load_ear_score(get_file_full())
    assert len(ear_l) == len(ear_r)
    result_df, result_threshold = blinking.peaks(ear_l, partial_threshold="auto")

    assert isinstance(result_df, pd.DataFrame)
    assert math.isclose(result_threshold, 0.18, abs_tol=0.01)

def test_peaks_with_invalid_input():
    time_series = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1], dtype=np.float16)
    invalid_time_series = [0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, "invalid"]
    invalid_minimum_distance = "invalid"
    invalid_minimum_prominence = -0.1
    invalid_minimum_internal_width = -10
    invalid_maximum_internal_width = -100
    invalid_partial_threshold = ("invalid",)
    
    with pytest.raises(TypeError):
        blinking.peaks(invalid_time_series)
    
    with pytest.raises(TypeError):
        blinking.peaks(time_series, minimum_distance=invalid_minimum_distance)
    
    with pytest.raises(ValueError):
        blinking.peaks(time_series, minimum_prominence=invalid_minimum_prominence)
    
    with pytest.raises(ValueError):
        blinking.peaks(time_series, minimum_internal_width=invalid_minimum_internal_width)
    
    with pytest.raises(ValueError):
        blinking.peaks(time_series, maximum_internal_width=invalid_maximum_internal_width)
    
    with pytest.raises(TypeError):
        blinking.peaks(time_series, partial_threshold=invalid_partial_threshold)
        
    with pytest.raises(ValueError):
        blinking.peaks(np.zeros((100, 3000)), partial_threshold="invalid")
        
    with pytest.raises(ValueError):
        blinking.peaks(np.full((3000), fill_value=np.nan), partial_threshold="auto")
        
    with pytest.raises(ValueError):
        blinking.peaks(time_series, minimum_distance=-1)
        
    with pytest.raises(TypeError):
        blinking.peaks(time_series, minimum_prominence="invalid")
        
    with pytest.raises(TypeError):
        blinking.peaks(time_series, minimum_internal_width="auto")
    
    with pytest.raises(ValueError):
        blinking.peaks(time_series, maximum_internal_width=100, minimum_internal_width=200)
        
    with pytest.raises(TypeError):
        blinking.peaks(time_series, maximum_internal_width="invalid")
        
    with pytest.raises(ValueError):
        blinking.peaks(time_series, partial_threshold="invalid")
        
    with pytest.raises(ValueError):
        blinking.peaks(time_series, partial_threshold=-0.1)
        


def test_blinking_io_load():
    with pytest.raises(FileNotFoundError):
        blinking.load_ear_score(Path("nonexistent.csv"))

    with pytest.raises(FileNotFoundError):
        blinking.load_ear_score(Path("nonexistent.csv"))
        
    with pytest.raises(FileNotFoundError):
        blinking.load_ear_score(get_file_short().parent)
        
    with pytest.raises(TypeError): # We do not support strings as paths
        blinking.load_ear_score("nonexistent.csv")
        
    with pytest.raises(TypeError):
        blinking.load_ear_score(1)
        
    with pytest.raises(ValueError):
        blinking.load_ear_score(get_file_short(), column_left="nonexistent")
        
    with pytest.raises(ValueError):
        blinking.load_ear_score(get_file_short(), column_right="nonexistent")
        
    with pytest.raises(ValueError):
        blinking.load_ear_score(get_file_short(), column_left="EAR2D6_l", column_right="EAR2D6_l")

    ear_l, ear_r = blinking.load_ear_score(get_file_short(), column_left="EAR2D6_l", column_right="EAR2D6_r")
    assert len(ear_l) == len(ear_r)
    assert isinstance(ear_l, np.ndarray)
    assert isinstance(ear_r, np.ndarray)

    # check that both are all not the same
    assert not np.all(ear_l == ear_r)
    
def test_blinking_extraction():
    ear_l, ear_r = blinking.load_ear_score(get_file_short())
    assert len(ear_l) == len(ear_r)
    
    # from the short_file we have knowledge about the peaks, and now we can check if the peaks are correctly extracted
    # all the testing is without smoothing
    assert isinstance(blinking.peaks(ear_l)[0], pd.DataFrame)
    assert isinstance(blinking.peaks(ear_r)[0], pd.DataFrame)
    
    assert math.isclose(blinking.peaks(ear_l, partial_threshold="auto")[1], 0.29, abs_tol=0.01)
    assert math.isclose(blinking.peaks(ear_r, partial_threshold="auto")[1], 0.29, abs_tol=0.01)
    
    # Test case 1: Default parameters
    threshold = 0.25
    dl, tl = blinking.peaks(ear_l, partial_threshold=threshold)
    dr, tr = blinking.peaks(ear_r, partial_threshold=threshold)
    assert tl == threshold and tr == threshold
    assert len(dl) == 7
    assert len(dr) == 7
    assert len(dl.columns) == 9 and len(dr.columns) == 9
    assert len(dl[dl["blink_type"] == "partial"])  == 2
    assert len(dl[dl["blink_type"] == "complete"]) == 5
    assert len(dr[dr["blink_type"] == "partial"])  == 1
    assert len(dr[dr["blink_type"] == "complete"]) == 6
    
    # Test case 2: Lower threshold
    threshold = 0.20
    dl, tl = blinking.peaks(ear_l, partial_threshold=threshold)
    dr, tr = blinking.peaks(ear_r, partial_threshold=threshold)
    assert tl == threshold and tr == threshold
    assert len(dl) == 7
    assert len(dr) == 7
    assert len(dl.columns) == 9 and len(dr.columns) == 9
    assert len(dl[dl["blink_type"] == "partial"])  == 2
    assert len(dl[dl["blink_type"] == "complete"]) == 5
    assert len(dr[dr["blink_type"] == "partial"])  == 1
    assert len(dr[dr["blink_type"] == "complete"]) == 6
    
    # Test case 3: Higher threshold
    threshold = 0.40
    dl, tl = blinking.peaks(ear_l, partial_threshold=threshold)
    dr, tr = blinking.peaks(ear_r, partial_threshold=threshold)
    assert tl == threshold and tr == threshold
    assert len(dl) == 7
    assert len(dr) == 7
    assert len(dl.columns) == 9 and len(dr.columns) == 9
    assert len(dl[dl["blink_type"] == "partial"])  == 7
    assert len(dl[dl["blink_type"] == "complete"]) == 0
    assert len(dr[dr["blink_type"] == "partial"])  == 7
    assert len(dr[dr["blink_type"] == "complete"]) == 0
    
    # Test case 4: Even lower threshold
    threshold = 0.10
    dl, tl = blinking.peaks(ear_l, partial_threshold=threshold)
    dr, tr = blinking.peaks(ear_r, partial_threshold=threshold)
    assert tl == threshold and tr == threshold
    assert len(dl) == 7
    assert len(dr) == 7
    assert len(dl.columns) == 9 and len(dr.columns) == 9
    assert len(dl[dl["blink_type"] == "partial"])  == 1
    assert len(dl[dl["blink_type"] == "complete"]) == 6
    assert len(dr[dr["blink_type"] == "partial"])  == 0
    assert len(dr[dr["blink_type"] == "complete"]) == 7
     
    # Test case 5: Even even lower threshold
    threshold = 0.05
    dl, tl = blinking.peaks(ear_l, partial_threshold=threshold)
    dr, tr = blinking.peaks(ear_r, partial_threshold=threshold)
    assert tl == threshold and tr == threshold
    assert len(dl) == 7
    assert len(dr) == 7
    assert len(dl.columns) == 9 and len(dr.columns) == 9
    assert len(dl[dl["blink_type"] == "partial"])  == 0
    assert len(dl[dl["blink_type"] == "complete"]) == 7
    assert len(dr[dr["blink_type"] == "partial"])  == 0
    assert len(dr[dr["blink_type"] == "complete"]) == 7
    
    # Test case 6: Even even even lower threshold
    threshold = 0.00
    dl, tl = blinking.peaks(ear_l, partial_threshold=threshold)
    dr, tr = blinking.peaks(ear_r, partial_threshold=threshold)
    assert tl == threshold and tr == threshold
    assert len(dl) == 7
    assert len(dr) == 7
    assert len(dl.columns) == 9 and len(dr.columns) == 9
    assert len(dl[dl["blink_type"] == "partial"])  == 0
    assert len(dl[dl["blink_type"] == "complete"]) == 7
    assert len(dr[dr["blink_type"] == "partial"])  == 0
    assert len(dr[dr["blink_type"] == "complete"]) == 7
    
    # Test case 7: Higher minimum distance
    threshold = 0.15
    minimum_distance = 4000
    dl, tl = blinking.peaks(ear_l, partial_threshold=threshold, minimum_distance=minimum_distance)
    dr, tr = blinking.peaks(ear_r, partial_threshold=threshold, minimum_distance=minimum_distance)
    assert tl == threshold and tr == threshold
    assert len(dl) == 3
    assert len(dr) == 3
    assert len(dl.columns) == 9 and len(dr.columns) == 9
    assert len(dl[dl["blink_type"] == "partial"])  == 0
    assert len(dl[dl["blink_type"] == "complete"]) == 3
    assert len(dr[dr["blink_type"] == "partial"])  == 0
    assert len(dr[dr["blink_type"] == "complete"]) == 3    
    
    # Test case 7: Higher minimum distance
    threshold = 0.15
    minimum_distance = 5000
    dl, tl = blinking.peaks(ear_l, partial_threshold=threshold, minimum_distance=minimum_distance)
    dr, tr = blinking.peaks(ear_r, partial_threshold=threshold, minimum_distance=minimum_distance)
    assert tl == threshold and tr == threshold
    assert len(dl) == 2
    assert len(dr) == 2
    assert len(dl.columns) == 9 and len(dr.columns) == 9
    assert len(dl[dl["blink_type"] == "partial"])  == 0
    assert len(dl[dl["blink_type"] == "complete"]) == 2
    assert len(dr[dr["blink_type"] == "partial"])  == 0
    assert len(dr[dr["blink_type"] == "complete"]) == 2
    
    # Test case 8: Higher minimum distance
    threshold = 0.15
    minimum_distance = 10_000
    dl, tl = blinking.peaks(ear_l, partial_threshold=threshold, minimum_distance=minimum_distance)
    dr, tr = blinking.peaks(ear_r, partial_threshold=threshold, minimum_distance=minimum_distance)
    assert tl == threshold and tr == threshold
    assert len(dl) == 2
    assert len(dr) == 2
    assert len(dl.columns) == 9 and len(dr.columns) == 9
    assert len(dl[dl["blink_type"] == "partial"])  == 1
    assert len(dl[dl["blink_type"] == "complete"]) == 1
    assert len(dr[dr["blink_type"] == "partial"])  == 1
    assert len(dr[dr["blink_type"] == "complete"]) == 1
    
    # Test case 8: Higher minimum distance than the length of the time series
    threshold = 0.15
    minimum_distance = 50_000
    dl, tl = blinking.peaks(ear_l, partial_threshold=threshold, minimum_distance=minimum_distance)
    dr, tr = blinking.peaks(ear_r, partial_threshold=threshold, minimum_distance=minimum_distance)
    assert tl == threshold and tr == threshold
    assert len(dl) == 1
    assert len(dr) == 1
    assert len(dl.columns) == 9 and len(dr.columns) == 9
    assert len(dl[dl["blink_type"] == "partial"])  == 0
    assert len(dl[dl["blink_type"] == "complete"]) == 1
    assert len(dr[dr["blink_type"] == "partial"])  == 0
    assert len(dr[dr["blink_type"] == "complete"]) == 1

    # Test case 9: Default minimum prominence
    threshold = 0.30
    minimum_prominence = 0.05
    dl, tl = blinking.peaks(ear_l, partial_threshold=threshold, minimum_prominence=minimum_prominence)
    dr, tr = blinking.peaks(ear_r, partial_threshold=threshold, minimum_prominence=minimum_prominence)
    assert tl == threshold and tr == threshold
    assert len(dl) == 7
    assert len(dr) == 7
    assert len(dl.columns) == 9 and len(dr.columns) == 9
    assert len(dl[dl["blink_type"] == "partial"])  == 5
    assert len(dl[dl["blink_type"] == "complete"]) == 2
    assert len(dr[dr["blink_type"] == "partial"])  == 4
    assert len(dr[dr["blink_type"] == "complete"]) == 3
    
    # Test case 10: Default minimum prominence
    threshold = 0.30
    minimum_prominence = 0.10
    dl, tl = blinking.peaks(ear_l, partial_threshold=threshold, minimum_prominence=minimum_prominence)
    dr, tr = blinking.peaks(ear_r, partial_threshold=threshold, minimum_prominence=minimum_prominence)
    assert tl == threshold and tr == threshold
    assert len(dl) == 6
    assert len(dr) == 7
    assert len(dl.columns) == 9 and len(dr.columns) == 9
    assert len(dl[dl["blink_type"] == "partial"])  == 4
    assert len(dl[dl["blink_type"] == "complete"]) == 2
    assert len(dr[dr["blink_type"] == "partial"])  == 4
    assert len(dr[dr["blink_type"] == "complete"]) == 3
    
    # Test case 11: Higher minimum prominence
    threshold = 0.30
    minimum_prominence = 0.2
    dl, tl = blinking.peaks(ear_l, partial_threshold=threshold, minimum_prominence=minimum_prominence)
    dr, tr = blinking.peaks(ear_r, partial_threshold=threshold, minimum_prominence=minimum_prominence)
    assert tl == threshold and tr == threshold
    assert len(dl) == 5
    assert len(dr) == 6
    assert len(dl.columns) == 9 and len(dr.columns) == 9
    assert len(dl[dl["blink_type"] == "partial"])  == 3
    assert len(dl[dl["blink_type"] == "complete"]) == 2
    assert len(dr[dr["blink_type"] == "partial"])  == 3
    assert len(dr[dr["blink_type"] == "complete"]) == 3
    
    # Test case 12: Higher minimum prominence
    threshold = 0.30
    minimum_prominence = 0.3
    dl, tl = blinking.peaks(ear_l, partial_threshold=threshold, minimum_prominence=minimum_prominence)
    dr, tr = blinking.peaks(ear_r, partial_threshold=threshold, minimum_prominence=minimum_prominence)
    assert tl == threshold and tr == threshold
    assert len(dl) == 2
    assert len(dr) == 3
    assert len(dl.columns) == 9 and len(dr.columns) == 9
    assert len(dl[dl["blink_type"] == "partial"])  == 0
    assert len(dl[dl["blink_type"] == "complete"]) == 2
    assert len(dr[dr["blink_type"] == "partial"])  == 0
    assert len(dr[dr["blink_type"] == "complete"]) == 3
    
    # Test case 13: Higher minimum internal width
    threshold = 0.30
    minimum_internal_width = 20
    dl, tl = blinking.peaks(ear_l, partial_threshold=threshold, minimum_internal_width=minimum_internal_width)
    dr, tr = blinking.peaks(ear_r, partial_threshold=threshold, minimum_internal_width=minimum_internal_width)
    assert tl == threshold and tr == threshold
    assert len(dl) == 6
    assert len(dr) == 7
    assert len(dl.columns) == 9 and len(dr.columns) == 9
    assert len(dl[dl["blink_type"] == "partial"])  == 4
    assert len(dl[dl["blink_type"] == "complete"]) == 2
    assert len(dr[dr["blink_type"] == "partial"])  == 4
    assert len(dr[dr["blink_type"] == "complete"]) == 3
    
    # Test case 14: Higher minimum internal width
    threshold = 0.30
    minimum_internal_width = 30
    dl, tl = blinking.peaks(ear_l, partial_threshold=threshold, minimum_internal_width=minimum_internal_width)
    dr, tr = blinking.peaks(ear_r, partial_threshold=threshold, minimum_internal_width=minimum_internal_width)
    assert tl == threshold and tr == threshold
    assert len(dl) == 3
    assert len(dr) == 4
    assert len(dl.columns) == 9 and len(dr.columns) == 9
    assert len(dl[dl["blink_type"] == "partial"])  == 2
    assert len(dl[dl["blink_type"] == "complete"]) == 1
    assert len(dr[dr["blink_type"] == "partial"])  == 1
    assert len(dr[dr["blink_type"] == "complete"]) == 3

    # Test case 15: Higher minimum internal width
    threshold = 0.30
    minimum_internal_width = 40
    dl, tl = blinking.peaks(ear_l, partial_threshold=threshold, minimum_internal_width=minimum_internal_width)
    dr, tr = blinking.peaks(ear_r, partial_threshold=threshold, minimum_internal_width=minimum_internal_width)
    assert tl == threshold and tr == threshold
    assert len(dl) == 1
    assert len(dr) == 1
    assert len(dl.columns) == 9 and len(dr.columns) == 9
    assert len(dl[dl["blink_type"] == "partial"])  == 0
    assert len(dl[dl["blink_type"] == "complete"]) == 1
    assert len(dr[dr["blink_type"] == "partial"])  == 0
    assert len(dr[dr["blink_type"] == "complete"]) == 1

    # Test case 16: Higher minimum internal width
    threshold = 0.30
    minimum_internal_width = 50
    dl, tl = blinking.peaks(ear_l, partial_threshold=threshold, minimum_internal_width=minimum_internal_width)
    dr, tr = blinking.peaks(ear_r, partial_threshold=threshold, minimum_internal_width=minimum_internal_width)
    assert tl == threshold and tr == threshold
    assert len(dl) == 0
    assert len(dr) == 0
    assert len(dl.columns) == 9 and len(dr.columns) == 9
    assert len(dl[dl["blink_type"] == "partial"])  == 0
    assert len(dl[dl["blink_type"] == "complete"]) == 0
    assert len(dr[dr["blink_type"] == "partial"])  == 0
    assert len(dr[dr["blink_type"] == "complete"]) == 0
    
    # Test case 17: Smaller maximal internal width
    threshold = 0.30
    maximum_internal_width = 20
    dl, tl = blinking.peaks(ear_l, partial_threshold=threshold, maximum_internal_width=maximum_internal_width)
    dr, tr = blinking.peaks(ear_r, partial_threshold=threshold, maximum_internal_width=maximum_internal_width)
    assert tl == threshold and tr == threshold
    assert len(dl) == 2
    assert len(dr) == 1
    assert len(dl.columns) == 9 and len(dr.columns) == 9
    assert len(dl[dl["blink_type"] == "partial"])  == 2
    assert len(dl[dl["blink_type"] == "complete"]) == 0
    assert len(dr[dr["blink_type"] == "partial"])  == 1
    assert len(dr[dr["blink_type"] == "complete"]) == 0
    

def test_blinking_extraction_smoothing():
    ear_l, ear_r = blinking.load_ear_score(get_file_full())
    ear_l = blinking.smooth(ear_l)
    ear_r = blinking.smooth(ear_r)
    assert len(ear_l) == len(ear_r)
    
    # from the short_file we have knowledge about the peaks, and now we can check if the peaks are correctly extracted
    # all the testing is without smoothing
    blinks_l, th_l = blinking.peaks(ear_l)
    blinks_r, th_l = blinking.peaks(ear_r)
    
    assert isinstance(blinks_l, pd.DataFrame)
    assert isinstance(blinks_r, pd.DataFrame)
    
    cols_l = blinks_l.columns
    assert "index" in cols_l
    assert "apex_frame" in cols_l
    assert "ear_score" in cols_l
    assert "intersection_point_lower" in cols_l
    assert "intersection_point_upper" in cols_l
    assert "prominance" in cols_l
    assert "peak_internal_width" in cols_l
    assert "peak_height" in cols_l
    
    cols_r = blinks_r.columns
    assert "index" in cols_r
    assert "apex_frame" in cols_r
    assert "ear_score" in cols_r
    assert "intersection_point_lower" in cols_r
    assert "intersection_point_upper" in cols_r
    assert "prominance" in cols_r
    assert "peak_internal_width" in cols_r
    assert "peak_height" in cols_r
    
    # this testing is less intesity, because the non-smoothed version is already tested and has a higher coverage
    # Test case 1: Default parameters
    threshold = 0.25    
    assert len(blinking.peaks(ear_l, partial_threshold=threshold)[0]) == 318
    assert len(blinking.peaks(ear_r, partial_threshold=threshold)[0]) == 327

    # Test case 2: Lower threshold
    threshold = 0.20
    assert len(blinking.peaks(ear_l, partial_threshold=threshold)[0]) == 318
    assert len(blinking.peaks(ear_r, partial_threshold=threshold)[0]) == 327
    
    # Test case 3: Higher threshold
    threshold = 0.40
    assert len(blinking.peaks(ear_l, partial_threshold=threshold)[0]) == 318
    assert len(blinking.peaks(ear_r, partial_threshold=threshold)[0]) == 327
    
    # Test case 4: Even lower threshold
    threshold = 0.10
    assert len(blinking.peaks(ear_l, partial_threshold=threshold)[0]) == 318
    assert len(blinking.peaks(ear_r, partial_threshold=threshold)[0]) == 327
    
    # Test case 5: Even even lower threshold
    threshold = 0.05
    assert len(blinking.peaks(ear_l, partial_threshold=threshold)[0]) == 318
    assert len(blinking.peaks(ear_r, partial_threshold=threshold)[0]) == 327
    
    # Test case 6: Even even even lower threshold
    threshold = 0.00
    assert len(blinking.peaks(ear_l, partial_threshold=threshold)[0]) == 318
    assert len(blinking.peaks(ear_r, partial_threshold=threshold)[0]) == 327
    
    # Test case 7: Higher minimum distance
    threshold = 0.15
    minimum_distance = 4000
    assert len(blinking.peaks(ear_l, partial_threshold=threshold, minimum_distance=minimum_distance)[0]) == 54
    assert len(blinking.peaks(ear_r, partial_threshold=threshold, minimum_distance=minimum_distance)[0]) == 54

    # Test case 7: Higher minimum distance
    threshold = 0.15
    minimum_distance = 5000
    assert len(blinking.peaks(ear_l, partial_threshold=threshold, minimum_distance=minimum_distance)[0]) == 42
    assert len(blinking.peaks(ear_r, partial_threshold=threshold, minimum_distance=minimum_distance)[0]) == 42
    
    # Test case 8: Higher minimum distance
    threshold = 0.15
    minimum_distance = 10_000
    assert len(blinking.peaks(ear_l, partial_threshold=threshold, minimum_distance=minimum_distance)[0]) == 21
    assert len(blinking.peaks(ear_r, partial_threshold=threshold, minimum_distance=minimum_distance)[0]) == 24
    
    # Test case 8: Higher minimum distance than the length of the time series
    threshold = 0.15
    minimum_distance = 50_000
    assert len(blinking.peaks(ear_l, partial_threshold=threshold, minimum_distance=minimum_distance)[0]) == 5
    assert len(blinking.peaks(ear_r, partial_threshold=threshold, minimum_distance=minimum_distance)[0]) == 5

    # Test case 9: Default minimum prominence
    threshold = 0.30
    minimum_prominence = 0.05
    assert len(blinking.peaks(ear_l, partial_threshold=threshold, minimum_prominence=minimum_prominence)[0]) == 318
    assert len(blinking.peaks(ear_r, partial_threshold=threshold, minimum_prominence=minimum_prominence)[0]) == 327
    
    # Test case 10: Default minimum prominence
    threshold = 0.30
    minimum_prominence = 0.10
    assert len(blinking.peaks(ear_l, partial_threshold=threshold, minimum_prominence=minimum_prominence)[0]) == 303
    assert len(blinking.peaks(ear_r, partial_threshold=threshold, minimum_prominence=minimum_prominence)[0]) == 309
    
    # Test case 11: Higher minimum prominence
    threshold = 0.30
    minimum_prominence = 0.2
    assert len(blinking.peaks(ear_l, partial_threshold=threshold, minimum_prominence=minimum_prominence)[0]) == 184
    assert len(blinking.peaks(ear_r, partial_threshold=threshold, minimum_prominence=minimum_prominence)[0]) == 249
    
    # Test case 12: Higher minimum prominence
    threshold = 0.30
    minimum_prominence = 0.3
    assert len(blinking.peaks(ear_l, partial_threshold=threshold, minimum_prominence=minimum_prominence)[0]) == 3
    assert len(blinking.peaks(ear_r, partial_threshold=threshold, minimum_prominence=minimum_prominence)[0]) == 5

    # Test case 13: Higher minimum internal width
    threshold = 0.30
    minimum_internal_width = 20
    assert len(blinking.peaks(ear_l, partial_threshold=threshold, minimum_internal_width=minimum_internal_width)[0]) == 318
    assert len(blinking.peaks(ear_r, partial_threshold=threshold, minimum_internal_width=minimum_internal_width)[0]) == 327
    
    # Test case 14: Higher minimum internal width
    threshold = 0.30
    minimum_internal_width = 30
    assert len(blinking.peaks(ear_l, partial_threshold=threshold, minimum_internal_width=minimum_internal_width)[0]) == 318
    assert len(blinking.peaks(ear_r, partial_threshold=threshold, minimum_internal_width=minimum_internal_width)[0]) == 325

    # Test case 15: Higher minimum internal width
    threshold = 0.30
    minimum_internal_width = 40
    assert len(blinking.peaks(ear_l, partial_threshold=threshold, minimum_internal_width=minimum_internal_width)[0]) == 77
    assert len(blinking.peaks(ear_r, partial_threshold=threshold, minimum_internal_width=minimum_internal_width)[0]) == 123
    
    # Test case 16: Higher minimum internal width
    threshold = 0.30
    minimum_internal_width = 50
    assert len(blinking.peaks(ear_l, partial_threshold=threshold, minimum_internal_width=minimum_internal_width)[0]) == 1
    assert len(blinking.peaks(ear_r, partial_threshold=threshold, minimum_internal_width=minimum_internal_width)[0]) == 4
    
    # Test case 17: Smaller maximal internal width
    threshold = 0.30
    maximum_internal_width = 20
    assert len(blinking.peaks(ear_l, partial_threshold=threshold, maximum_internal_width=maximum_internal_width)[0]) == 0
    assert len(blinking.peaks(ear_r, partial_threshold=threshold, maximum_internal_width=maximum_internal_width)[0]) == 0

def test_blinking_extraction_full():
    ear_l, ear_r = blinking.load_ear_score(get_file_full())
    assert len(ear_l) == len(ear_r)
    
    total_left  = 340
    total_right = 354
    
    # Test case 1: Default parameters
    threshold = 0.25
    dl, tl = blinking.peaks(ear_l, partial_threshold=threshold)
    dr, tr = blinking.peaks(ear_r, partial_threshold=threshold)
    assert tl == threshold and tr == threshold
    assert len(dl) == total_left
    assert len(dr) == total_right
    assert len(dl.columns) == 9 and len(dr.columns) == 9
    assert len(dl[dl["blink_type"] == "partial"])  == 149
    assert len(dl[dl["blink_type"] == "complete"]) == 191
    assert len(dr[dr["blink_type"] == "partial"])  == 100
    assert len(dr[dr["blink_type"] == "complete"]) == 254
    

    # Test case 2: Lower threshold
    threshold = 0.20
    dl, tl = blinking.peaks(ear_l, partial_threshold=threshold)
    dr, tr = blinking.peaks(ear_r, partial_threshold=threshold)
    assert tl == threshold and tr == threshold
    assert len(dl) == total_left
    assert len(dr) == total_right
    assert len(dl.columns) == 9 and len(dr.columns) == 9
    assert len(dl[dl["blink_type"] == "partial"])  == 56
    assert len(dl[dl["blink_type"] == "complete"]) == 284
    assert len(dr[dr["blink_type"] == "partial"])  == 57
    assert len(dr[dr["blink_type"] == "complete"]) == 297
    
    # Test case 3: Higher threshold
    threshold = 0.40
    dl, tl = blinking.peaks(ear_l, partial_threshold=threshold)
    dr, tr = blinking.peaks(ear_r, partial_threshold=threshold)
    assert tl == threshold and tr == threshold
    assert len(dl) == total_left
    assert len(dr) == total_right
    assert len(dl.columns) == 9 and len(dr.columns) == 9
    assert len(dl[dl["blink_type"] == "partial"])  == 340
    assert len(dl[dl["blink_type"] == "complete"]) == 0
    assert len(dr[dr["blink_type"] == "partial"])  == 354
    assert len(dr[dr["blink_type"] == "complete"]) == 0
    
    # Test case 3: auto
    threshold = "auto"
    dl, tl = blinking.peaks(ear_l, partial_threshold=threshold)
    dr, tr = blinking.peaks(ear_r, partial_threshold=threshold)
    assert math.isclose(tl, 0.18, abs_tol=0.01)
    assert math.isclose(tr, 0.18, abs_tol=0.01)
    
    assert len(dl) == total_left
    assert len(dr) == total_right
    assert len(dl.columns) == 9 and len(dr.columns) == 9
    assert len(dl[dl["blink_type"] == "partial"])  == 48
    assert len(dl[dl["blink_type"] == "complete"]) == 292
    assert len(dr[dr["blink_type"] == "partial"])  == 55
    assert len(dr[dr["blink_type"] == "complete"]) == 299

    
def test_blinking_match_inputs():
    ear_l, ear_r = blinking.load_ear_score(get_file_short())
    assert len(ear_l) == len(ear_r)
    
    # Test case 1: Default parameters
    threshold = 0.18
    blinks_l, th_l = blinking.peaks(ear_l, partial_threshold=threshold)
    blinks_r, th_r = blinking.peaks(ear_r, partial_threshold=threshold)

    with pytest.raises(ValueError):
        blinking.match(None, blinks_r, tolerance=30)
    
    with pytest.raises(ValueError):
        blinking.match(blinks_l, None, tolerance=30)
    
    with pytest.raises(TypeError):
        blinking.match(1, blinks_r, tolerance=30)
    
    with pytest.raises(TypeError):
        blinking.match(blinks_l, 1, tolerance=30)
    
    with pytest.raises(TypeError):
        blinking.match("1", blinks_r, tolerance=30)
    
    with pytest.raises(TypeError):
        blinking.match(blinks_l, "1", tolerance=30)
    
    with pytest.raises(ValueError):
        blinking.match(pd.DataFrame(), blinks_r, tolerance=30)
    
    with pytest.raises(ValueError):
        blinking.match(blinks_l, pd.DataFrame(), tolerance=30)
    
    with pytest.raises(ValueError):
        blinking.match(blinks_l, blinks_r, tolerance=-1)
        
    blinks_l.drop(columns=["index"], inplace=True)
    blinks_r.drop(columns=["index"], inplace=True)
    
    blinking.match(blinks_l, blinks_r, tolerance=30)
    
    blink_l_t = blinks_l.drop(columns=["apex_frame"])
    with pytest.raises(ValueError):
        blinking.match(blink_l_t, blinks_r, tolerance=30)

    blink_r_t = blinks_r.drop(columns=["apex_frame"])
    with pytest.raises(ValueError):
        blinking.match(blinks_l, blink_r_t, tolerance=30)
        
    
def test_blinking_match_values():
    ear_l, ear_r = blinking.load_ear_score(get_file_short())
    assert len(ear_l) == len(ear_r)
    
    # Test case 1: Default parameters
    threshold = 0.30
    blinks_l, th_l = blinking.peaks(ear_l, partial_threshold=threshold)
    blinks_r, th_r = blinking.peaks(ear_r, partial_threshold=threshold)
    assert len(blinks_l) == 7
    assert len(blinks_r) == 7
    
    matched = blinking.match(blinks_l, blinks_r, tolerance=30)
    assert len(matched) == 7
    matched = blinking.match(blinks_l, blinks_r, tolerance=10)
    assert len(matched) == 7
    matched = blinking.match(blinks_l, blinks_r, tolerance=0)
    assert len(matched) == 13

def test_blinking_summary():
    ear_l, ear_r = blinking.load_ear_score(get_file_short())
    assert len(ear_l) == len(ear_r)
    
    # Test case 1: Default parameters
    threshold = 0.23
    blinks_l, th_l = blinking.peaks(ear_l, partial_threshold=threshold)
    blinks_r, th_r = blinking.peaks(ear_r, partial_threshold=threshold)
    matched = blinking.match(blinks_l, blinks_r, tolerance=30)
    summary = blinking.summarize(ear_l=ear_l, ear_r=ear_r, matched_blinks=matched, partial_threshold_l=th_l, partial_threshold_r=th_r)
    
    assert isinstance(summary, pd.DataFrame)
    
    ear_l, ear_r = blinking.load_ear_score(get_file_full())
    assert len(ear_l) == len(ear_r)
    
    # Test case 1: Default parameters
    threshold = 0.18
    blinks_l, th_l = blinking.peaks(ear_l, partial_threshold=threshold)
    blinks_r, th_r = blinking.peaks(ear_r, partial_threshold=threshold)
    matched = blinking.match(blinks_l, blinks_r, tolerance=30)
    summary = blinking.summarize(ear_l=ear_l, ear_r=ear_r, matched_blinks=matched, partial_threshold_l=th_l, partial_threshold_r=th_r)
    
    assert isinstance(summary, pd.DataFrame)

def test_blinking_summary_visual():
    ear_l, ear_r = blinking.load_ear_score(get_file_short())
    assert len(ear_l) == len(ear_r)
    
    # Test case 1: Default parameters
    threshold = 0.23
    blinks_l, th_l = blinking.peaks(ear_l, partial_threshold=threshold)
    blinks_r, th_r = blinking.peaks(ear_r, partial_threshold=threshold)
    matched = blinking.match(blinks_l, blinks_r, tolerance=30)
    summary = blinking.visualize(matched_df=matched)
    assert isinstance(summary, np.ndarray)

    ear_l, ear_r = blinking.load_ear_score(get_file_full())
    assert len(ear_l) == len(ear_r)
    
    # Test case 1: Default parameters
    threshold = 0.18
    blinks_l, th_l = blinking.peaks(ear_l, partial_threshold=threshold)
    blinks_r, th_r = blinking.peaks(ear_r, partial_threshold=threshold)
    matched = blinking.match(blinks_l, blinks_r, tolerance=30)
    summary = blinking.visualize(matched_df=matched)
    assert isinstance(summary, np.ndarray)
    
     
    # Test case 1: Default parameters
    threshold = 0.18
    blinks_l, th_l = blinking.peaks(ear_l, partial_threshold=threshold)
    blinks_r, th_r = blinking.peaks(ear_r, partial_threshold=threshold)
    matched = blinking.match(blinks_l, blinks_r, tolerance=30)
    
    with pytest.raises(ValueError):
        summary = blinking.visualize(matched_df=matched, mode="nonexistent")
 

@pytest.mark.usefixtures("clean_folder")
def test_blinking_io_save_excel(): 
    ear_l, ear_r = blinking.load_ear_score(get_file_full())
    assert len(ear_l) == len(ear_r)
    
    # Test case 1: Default parameters
    threshold = 0.30
    blinks_l, th_l = blinking.peaks(ear_l, partial_threshold=threshold)
    blinks_r, th_r = blinking.peaks(ear_r, partial_threshold=threshold)
    matched = blinking.match(blinks_l, blinks_r, tolerance=30)
    summary = blinking.summarize(ear_l=ear_l, ear_r=ear_r, matched_blinks=matched, partial_threshold_l=th_l, partial_threshold_r=th_r)
    
    assert isinstance(summary, pd.DataFrame)
    
    # now we can test the saving
    folder = get_temp_folder()
    folder.mkdir()
    test_files = folder / "test_file.csv"
    
    with pytest.raises(RuntimeError):
        blinking.save_results(test_files)
      
    with pytest.raises(RuntimeError):
        blinking.save_results(test_files.as_posix())  
        
    with pytest.raises(ValueError):
        blinking.save_results(test_files, blinks_l, format="nonexistent")
        
    with pytest.raises(ValueError):
        blinking.save_results(test_files.as_posix(), blinks_l, format="spss")
        
    blinking.save_results(test_files, blinks_l, format="excel")
    assert len(list(folder.iterdir())) == 1
    
    blinking.save_results(test_files, blinks_l, format="excel", exists_ok=True)
    assert len(list(folder.iterdir())) == 1
    
    with pytest.raises(FileExistsError):
        blinking.save_results(test_files, blinks_l, format="excel", exists_ok=False)
        
    blinking.save_results(test_files, blinks_l, blinks_r, matched, summary, format="excel", exists_ok=True)
    assert len(list(folder.iterdir())) == 1
    
    
@pytest.mark.usefixtures("clean_folder")
def test_blinking_io_save_csv(): 
    ear_l, ear_r = blinking.load_ear_score(get_file_full())
    assert len(ear_l) == len(ear_r)
    
    # Test case 1: Default parameters
    threshold = 0.30
    blinks_l, th_l = blinking.peaks(ear_l, partial_threshold=threshold)
    blinks_r, th_r = blinking.peaks(ear_r, partial_threshold=threshold)
    matched = blinking.match(blinks_l, blinks_r, tolerance=30)
    summary = blinking.summarize(ear_l=ear_l, ear_r=ear_r, matched_blinks=matched, partial_threshold_l=th_l, partial_threshold_r=th_r)
    
    assert isinstance(summary, pd.DataFrame)
    
    # now we can test the saving
    folder = get_temp_folder()
    folder.mkdir()
    test_files = folder / "test_file.csv"
    
    with pytest.raises(RuntimeError):
        blinking.save_results(test_files)
      
    with pytest.raises(RuntimeError):
        blinking.save_results(test_files.as_posix())  
        
    with pytest.raises(ValueError):
        blinking.save_results(test_files, blinks_l, format="nonexistent")
        
    with pytest.raises(ValueError):
        blinking.save_results(test_files.as_posix(), blinks_l, format="spss")
        
    blinking.save_results(test_files, blinks_l, format="csv")
    assert len(list(folder.iterdir())) == 1
    
    blinking.save_results(test_files, blinks_l, format="csv", exists_ok=True)
    assert len(list(folder.iterdir())) == 1
  
    blinking.save_results(test_files, blinks_l, blinks_r, matched, summary, format="csv", exists_ok=True)
    assert len(list(folder.iterdir())) == 4
    