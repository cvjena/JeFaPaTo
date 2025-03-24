__all__ = ["peaks", "peaks_espbm"]

from typing import Callable

import espbm
import numpy as np
import pandas as pd
from scipy import signal


def otsu_thresholding(values: np.ndarray) -> float:
    """
    Thresholding of the histogram values using the Otsu method.

    Computes the best threshold for the given values using the Otsu method splitting the values into two classes.
    The threshold is the value that minimizes the within-class variance.

    Based on https://en.wikipedia.org/wiki/Otsu%27s_method
    """
    # first remove all nans from the values
    values = values[~np.isnan(values)]

    # check that there are enough values for the calculation
    if len(values) < 6:
        return np.nan

    th_range = np.sort(np.unique(values))[3:-3]  # remove the first and last 3 values to avoid errors in the calculation
    res = []
    for th in th_range:
        r = np.nansum([np.mean(cls) * np.var(values, where=cls) for cls in [values >= th, values < th]])
        res.append(r)

    if len(res) == 0:
        # if the result is empty, return np.nan
        return np.nan
    return th_range[np.argmin(res)]


def peaks(
    time_series: np.ndarray,
    minimum_distance: int = 50,
    minimum_prominence: float = 0.05,
    minimum_internal_width: int = 10,
    maximum_internal_width: int = 250,
    partial_threshold: str | float = "auto",
) -> tuple[pd.DataFrame, float]:
    """
    Blinks the peaks of the EAR Score time series for a single eye.

    Args
    ----
    time_series : np.ndarray
        The input time series data.
    threshold : float
        The threshold value for filtering peaks.
    minimum_distance int, optional:
        The minimum distance between peaks. Defaults to 150.
    minimum_prominence : float, optional
        The minimum prominence of peaks. Defaults to 0.05.
    minimum_interal_width int, optional
        The minimum width of peaks. Defaults to 10.
    maximum_interal_width int, optional
        The maximum width of peaks. Defaults to 150.
    partial_threshold : str | float, optional
        The thresholding parameter for the partial blinks. Defaults to "auto".
        If "auto", the threshold is computed using the Otsu method.
        If a float, the threshold is set to the given value.

    Returns
    -------
    pd.DataFrame:
        A DataFrame containing information about the detected peaks.
        Columns: ["index", "apex_frame", "ear_score", "intersection_point_lower", "intersection_point_upper", "prominance", "peak_internal_width", "peak_height"]
    float
        The threshold used for partial blinks.

    Raises
    ------
    Value Error
        - If the thresholding parameter is a tuple, but does not have two values.
        - If the thresholding parameter is a tuple, but has negative values.
        - If the thresholding parameter is not a string or tuple.

    """
    if not isinstance(time_series, np.ndarray):
        raise TypeError("Time series is not a numpy array.")
    if time_series.ndim != 1:
        raise ValueError("Time series is not one-dimensional.")
    if np.isnan(time_series).any():
        raise ValueError("Time series contains NaN values.")

    if not isinstance(minimum_distance, int):
        raise TypeError("Minimum distance is not an integer.")
    if minimum_distance < 0:
        raise ValueError("Minimum distance is negative.")
    if not isinstance(minimum_prominence, float):
        raise TypeError("Minimum prominence is not a float.")
    if minimum_prominence < 0:
        raise ValueError("Minimum prominence is negative.")
    if not isinstance(minimum_internal_width, int):
        raise TypeError("Minimum internal width is not an integer.")
    if minimum_internal_width < 0:
        raise ValueError("Minimum internal width is negative.")
    if not isinstance(maximum_internal_width, int):
        raise TypeError("Maximum internal width is not an integer.")
    if maximum_internal_width < 0:
        raise ValueError("Maximum internal width is negative.")
    if maximum_internal_width < minimum_internal_width:
        raise ValueError("Maximum internal width is smaller than minimum internal width.")

    # check thresholding parameter
    if not (isinstance(partial_threshold, str) or isinstance(partial_threshold, float)):
        raise TypeError("Thresholding parameter is not a string or tuple.")
    if isinstance(partial_threshold, str):
        if partial_threshold != "auto":
            raise ValueError("Thresholding parameter is a string, but not 'auto'.")
    if isinstance(partial_threshold, float):
        if partial_threshold < 0:
            raise ValueError("Thresholding parameter is a float, but negative.")

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

    df = pd.DataFrame(blinks, columns=list(blinks.keys()), index=blinks["index"]).reset_index(drop=True)

    prominance = df["prominance"].to_numpy()
    df["blink_type"] = "none"

    # either estimate the threshold or use the given value
    th = otsu_thresholding(prominance) if partial_threshold == "auto" else partial_threshold

    # if the threshold is np.nan, then the thresholding failed
    # so set all blinks to complete
    if th is np.nan:
        for index, row in df.iterrows():
            df.loc[index, "blink_type"] = "complete"

        return df, th

    # set the blink type based on the threshold
    for index, row in df.iterrows():
        if row["prominance"] > th:
            df.loc[index, "blink_type"] = "complete"
        else:
            df.loc[index, "blink_type"] = "partial"
    return df, th


def peaks_espbm(
    time_series: np.ndarray,
    minimum_prominence: float = 0.05,
    partial_threshold: str | float = "auto",
    f_min: Callable | None = None,
    f_max: Callable | None = None,
    f_val: Callable | None = None,
) -> tuple[pd.DataFrame, float]:
    prototype, params = espbm.manual.define_prototype(return_params=True)
    matches = espbm.match.find_prototype(time_series, prototype, max_prototype_distance=3.0)

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

    if f_min is not None:
        f_min(0)
    if f_max is not None:
        f_max(len(matches))
    if f_val is not None:
        f_val(0)

    for idx, (_from, _to, _) in enumerate(matches):
        if f_val is not None:
            f_val(idx)

        interval = time_series[_from:_to]
        _, o_params = espbm.match.optim(interval=interval, prototype_params=params)
        if o_params is None:  # if the optimization failed, skip the blink
            continue
        props = espbm.match.interval_stats(interval, o_params)

        prominance = props["prominance"]
        if prominance < minimum_prominence:
            continue

        intersection_point_left = props["ips_left"]
        intersection_point_right = props["ips_right"]
        peak_height = props["heights"]
        peak_interal_width = props["internal_width"]

        blinks["index"].append(idx)
        blinks["apex_frame"].append(props["apex_location"] + _from)
        blinks["ear_score"].append(props["apex_score"])

        blinks["intersection_point_lower"].append(intersection_point_left + _from)
        blinks["intersection_point_upper"].append(intersection_point_right + _from)

        blinks["prominance"].append(prominance)
        blinks["peak_internal_width"].append(peak_interal_width)
        blinks["peak_height"].append(peak_height)

    df = pd.DataFrame(blinks, columns=list(blinks.keys()), index=blinks["index"]).reset_index(drop=True)

    prominance = df["prominance"].to_numpy()
    df["blink_type"] = "none"

    # either estimate the threshold or use the given value
    th = otsu_thresholding(prominance) if partial_threshold == "auto" else partial_threshold

    # if the threshold is np.nan, then the thresholding failed
    # so set all blinks to complete
    if th is np.nan:
        for index, row in df.iterrows():
            df.loc[index, "blink_type"] = "complete"

        return df, th

    # set the blink type based on the threshold
    for index, row in df.iterrows():
        if row["prominance"] > th:
            df.loc[index, "blink_type"] = "complete"
        else:
            df.loc[index, "blink_type"] = "partial"
    return df, th

    return matches
