__all__ = ["peaks"]

import numpy as np
import pandas as pd
from scipy import signal


def peaks(
    time_series: np.ndarray, 
    threshold: float = 0.25, 
    minimum_distance: int = 50,
    minimum_prominence: float = 0.05,
    minimum_internal_width: int = 10,
    maximum_internal_width: int = 250,
) -> pd.DataFrame:
    """
    Blinks the peaks of the EAR Score time series for a single eye.
    
    Args:
        time_series (np.ndarray): The input time series data.
        threshold (float): The threshold value for filtering peaks.
        minimum_distance (int, optional): The minimum distance between peaks. Defaults to 150.
        minimum_prominence (float, optional): The minimum prominence of peaks. Defaults to 0.05.
        minimum_interal_width (int, optional): The minimum width of peaks. Defaults to 10.
        maximum_interal_width (int, optional): The maximum width of peaks. Defaults to 150.
    
    Returns:
        pd.DataFrame: A DataFrame containing information about the detected peaks.
            Columns: ["index", "apex_frame", "ear_score", "intersection_point_lower", "intersection_point_upper", "prominance", "peak_internal_width", "peak_height"]
    """
    # Find the peaks by turning the time series upside down
    peaks, props = signal.find_peaks(
        x=-time_series,
        distance=minimum_distance, 
        prominence=minimum_prominence,
        width=minimum_internal_width,
    )

    blinks = {
        "index": [],
        "apex_frame": [],
        "ear_score": [],
        "intersection_point_lower": [],
        "intersection_point_upper": [],
        "prominance": [],
        "peak_internal_width": [],
        "peak_height": [],
    }
    time_series = time_series.round(4)
    for idx, peak in enumerate(peaks):
        if time_series[peak] > threshold:
            continue

        prominance = props["prominences"][idx].round(4)
        intersection_point_left = props["left_ips"][idx].astype(np.int32)
        intersection_point_right = props["right_ips"][idx].astype(np.int32)
        peak_height = -props["width_heights"][idx].round(4)
        peak_interal_width = intersection_point_right - intersection_point_left

        if peak_interal_width > maximum_internal_width:
            continue

        blinks["index"].append(idx)
        blinks["apex_frame"].append(peak)
        blinks["ear_score"].append(time_series[peak])
        blinks["intersection_point_lower"].append(intersection_point_left)
        blinks["intersection_point_upper"].append(intersection_point_right)
        blinks["prominance"].append(prominance)
        blinks["peak_internal_width"].append(peak_interal_width)
        blinks["peak_height"].append(peak_height)

    return  pd.DataFrame(blinks, columns=list(blinks.keys()), index=blinks["index"]).reset_index(drop=True)
