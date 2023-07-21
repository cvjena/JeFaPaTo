__all__ = ["peaks"]

import numpy as np
import pandas as pd
from scipy import signal


def peaks(time_series: np.ndarray, threshold: float, **kwargs) -> pd.DataFrame:
    """
    Blinks the peaks of the LED.
    """
    distance = kwargs.get("distance", 150)
    prominence = kwargs.get("prominence", 0.05)
    width_min = kwargs.get("width_min", 10)
    width_max = kwargs.get("width_max", 150)

    peaks, props = signal.find_peaks(
        -time_series, distance=distance, prominence=prominence, width=width_min
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

        if widt > width_max:
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
