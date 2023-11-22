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
            Columns: ['index', 'frame', 'score', 'ips_l', 'ips_r', 'promi', 'width', 'height']
    """
    peaks, props = signal.find_peaks(
        x=-time_series, # invert the time series to find the peaks
        distance=minimum_distance, 
        prominence=minimum_prominence,
        width=minimum_internal_width,
    )

    blinkings = {
        "index": [],
        "frame": [],
        "score": [],
        "ips_l": [],
        "ips_r": [],
        "promi": [],
        "width": [],
        "height": [],
    }
    time_series = time_series.round(4)
    for idx, peak in enumerate(peaks):
        if time_series[peak] > threshold:
            continue

        prom = props["prominences"][idx].round(4)
        ipsl = props["left_ips"][idx].astype(np.int32)
        ipsr = props["right_ips"][idx].astype(np.int32)
        whei = -props["width_heights"][idx].round(4)
        widt = ipsr - ipsl

        if widt > maximum_internal_width:
            continue

        blinkings["index"].append(idx)
        blinkings["frame"].append(peak)
        blinkings["score"].append(time_series[peak])
        blinkings["ips_l"].append(ipsl)
        blinkings["ips_r"].append(ipsr)
        blinkings["promi"].append(prom)
        blinkings["width"].append(widt)
        blinkings["height"].append(whei)

    return  pd.DataFrame(blinkings, columns=list(blinkings.keys()), index=blinkings["index"]).reset_index(drop=True)
