__all__ = ["summarize", "visualize"]

import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_blink_start(df: pd.Series | pd.DataFrame, blink_id: int) -> int:
    """
    Find the frame number of the start of the blink based on the blink_id.
    """
    assert isinstance(df, pd.Series) or isinstance(df, pd.DataFrame), "df must be a pandas Series or DataFrame"
    assert isinstance(blink_id, int), "blink_id must be an integer"

    # first remove all nan rows
    temp_df = df.dropna(how="any", inplace=False)  # do not modify the original DataFrame!
    temp_df = temp_df.reset_index(drop=True)
    # find the row with the blink_id
    if blink_id >= len(temp_df):
        raise ValueError(f"blink_id {blink_id} is outside the range of the DataFrame (0-{len(temp_df) - 1})")
    row = temp_df.iloc[blink_id]

    # get the apex frame and the width
    start_value = row["apex_frame_og"] - row["peak_internal_width"] // 2
    start_value = int(start_value)
    return start_value


def summarize(
    ear_l: np.ndarray,
    ear_r: np.ndarray,
    matched_blinks: pd.DataFrame,
    partial_threshold_l: float,
    partial_threshold_r: float,
    fps: int = 240,
) -> pd.DataFrame:
    """
    Summarizes the matched blinks data by computing statistics for each minute.
    Based on the requirements of the medical study, the statistics are computed for the width and height of the blinks.

    Possible extensions:
        - compute the statistics for the distance between the left and right eye
        - compute the statistics for the duration of the blinks
        - compute the statistics for the velocity of the blinks
        - compute the statistics for the acceleration of the blinks

    Parameters
    ----------
    ear_l (np.ndarray):
        Array containing the left eye EAR values.
    ear_r (np.ndarray):
        Array containing the right eye EAR values.
    matched_blinks (pd.DataFrame):
        DataFrame containing the matched blinks data.
    fps (int):
        Frames per second. Default is 240.
    partial_threshold_l (float):
        Threshold for the left eye. If the EAR value is below this threshold, the blink is considered partial.
    partial_threshold_r (float):
        Threshold for the right eye. If the EAR value is below this threshold, the blink is considered partial.

    Returns
    --------
        pd.DataFrame: DataFrame containing the summarized statistics
    """
    matched_blinks = pd.DataFrame(matched_blinks, copy=True)
    # matched_blinks.dropna(subset=[("left", "blink_type"), ("right", "blink_type")], inplace=True)

    # compute the statistics which the medical partners are interested in
    length_l_min = len(ear_l) / fps / 60
    length_r_min = len(ear_r) / fps / 60

    statistics: dict[str, Any] = {}

    # average ear score 3 seconds before the first blink
    end_l = get_blink_start(matched_blinks["left"], 0)  # call it end even though it is the start, because it is the end of the 3 seconds
    end_r = get_blink_start(matched_blinks["right"], 0)
    start_l = max(end_l - 3 * fps, 0)
    start_r = max(end_r - 3 * fps, 0)

    statistics["EAR_Before_Blink_left_avg"] = np.nanmean(ear_l[start_l:end_l])
    statistics["EAR_Before_Blink_right_avg"] = np.nanmean(ear_r[start_r:end_r])

    statistics["EAR_left_min"] = np.nanmin(ear_l)
    statistics["EAR_right_min"] = np.nanmin(ear_r)
    statistics["EAR_left_max"] = np.nanmax(ear_l)
    statistics["EAR_right_max"] = np.nanmax(ear_r)

    statistics["Partial_Blink_threshold_left"] = partial_threshold_l
    statistics["Partial_Blink_threshold_right"] = partial_threshold_r

    # get all prominances
    prom_l = matched_blinks["left"]["prominance"]
    prom_r = matched_blinks["right"]["prominance"]
    prom = np.concatenate([prom_l, prom_r])

    statistics["Prominance_min"] = np.nanmin(prom)
    statistics["Prominance_max"] = np.nanmax(prom)
    statistics["Prominance_avg"] = np.nanmean(prom)

    # get all widths
    width_l = matched_blinks["left"]["peak_internal_width"]
    width_r = matched_blinks["right"]["peak_internal_width"]
    width = np.concatenate([width_l, width_r])

    statistics["Width_min"] = np.nanmin(width)
    statistics["Width_max"] = np.nanmax(width)
    statistics["Width_avg"] = np.nanmean(width)

    # get all heights
    height_l = matched_blinks["left"]["peak_height"]
    height_r = matched_blinks["right"]["peak_height"]
    height = np.concatenate([height_l, height_r])

    statistics["Height_min"] = np.nanmin(height)
    statistics["Height_max"] = np.nanmax(height)
    statistics["Height_avg"] = np.nanmean(height)

    # partial blinks counting
    partial_l = matched_blinks["left"][matched_blinks["left"]["blink_type"] == "partial"]
    partial_r = matched_blinks["right"][matched_blinks["right"]["blink_type"] == "partial"]

    statistics["Partial_Blink_Total_left"] = len(partial_l)
    statistics["Partial_Blink_Total_right"] = len(partial_r)
    statistics["Partial_Frequency_left_bpm"] = statistics["Partial_Blink_Total_left"] / length_l_min
    statistics["Partial_Frequency_right_bpm"] = statistics["Partial_Blink_Total_right"] / length_r_min

    # blink lengths in ms
    blink_lengths_l_ms = matched_blinks["left"]["peak_internal_width"] * 1000 / fps  # in ms
    blink_lengths_r_ms = matched_blinks["right"]["peak_internal_width"] * 1000 / fps  # in ms

    statistics["Blink_Length_left_ms_avg"] = np.nanmean(blink_lengths_l_ms)
    statistics["Blink_Length_left_ms_std"] = np.nanstd(blink_lengths_l_ms)
    statistics["Blink_Length_right_ms_avg"] = np.nanmean(blink_lengths_r_ms)
    statistics["Blink_Length_right_ms_std"] = np.nanstd(blink_lengths_r_ms)

    statistics.update(blinks_per_min(partial_l, length_l_min, fps, "left", "Partial"))
    statistics.update(blinks_per_min(partial_r, length_r_min, fps, "right", "Partial"))

    # complete blinks counting
    complete_l = matched_blinks["left"][matched_blinks["left"]["blink_type"] == "complete"]
    complete_r = matched_blinks["right"][matched_blinks["right"]["blink_type"] == "complete"]

    statistics["Complete_Blink_Total_left"] = len(complete_l)
    statistics["Complete_Blink_Total_right"] = len(complete_r)
    statistics["Complete_Frequency_left_bpm"] = statistics["Complete_Blink_Total_left"] / length_l_min
    statistics["Complete_Frequency_right_bpm"] = statistics["Complete_Blink_Total_right"] / length_r_min

    statistics.update(blinks_per_min(complete_l, length_l_min, fps, "left", "Complete"))
    statistics.update(blinks_per_min(complete_r, length_r_min, fps, "right", "Complete"))

    out_df = pd.DataFrame.from_dict(statistics, orient="index")
    # make the index a column
    out_df.reset_index(inplace=True)
    out_df.columns = ["statistics", "value"]
    # compute the statistics for the left eye
    return out_df


def blinks_per_min(selection: pd.DataFrame | pd.Series, length: float, fps: int, side: str, prefix: str) -> dict[str, int]:
    """
    Calculate the number of blinks per minute from a given selection of blink data.
    Args:
        selection (pd.DataFrame | pd.Series): A pandas DataFrame or Series containing blink data. Must include the 'apex_frame_og' column.
        length (float): The total length of the video in minutes.
        fps (int): Frames per second of the video.
        side (str): The side of the face being analyzed (e.g., 'left' or 'right').
        prefix (str): A prefix to be added to the keys in the resulting dictionary.
    Returns:
        dict[str, int]: A dictionary where keys are formatted as '{prefix}_Blinks_min{minute}_{side}' and values are the number of blinks in each minute.
    """
    stat: dict[str, int] = {}

    assert isinstance(selection, pd.DataFrame) or isinstance(selection, pd.Series), "selection must be a pandas DataFrame or Series"
    assert "apex_frame_og" in selection.columns, "selection must contain the 'apex_frame_og' column"

    selection["minute"] = selection["apex_frame_og"] / fps / 60
    partial_times_l: pd.Series = pd.to_datetime(selection["minute"], unit="m", errors="ignore")  # type: ignore
    selection_group = selection.groupby(partial_times_l.dt.minute)
    series = selection_group.count()["minute"]

    for i in range(0, math.ceil(length)):
        stat[f"{prefix}_Blinks_min{i + 1:02d}_{side}"] = series.get(key=i, default=0)  # type: ignore

    return stat


def visualize(
    matched_df: pd.DataFrame,
    fps: int = 240,
    rolling_mean: int = 5,
    mode: str = "complete",
) -> np.ndarray:
    """
    Visualizes the data from the matched_df DataFrame by creating a scatter plot
    of the time difference between left and right frames, along with rolling average
    and standard deviation. It also generates a histogram of the time differences
    and a bar chart showing the number of blinks per minute.

    Parameters:
        matched_df (pd.DataFrame): DataFrame containing the matched data.
        fps (int): Frames per second of the video. Default is 240.
        rolling_mean (int): Number of data points to use for computing rolling average
            and standard deviation. Default is 5.

    Returns:
        np.ndarray: RGBA array representing the generated plot.
    """
    if mode not in ["complete", "partial"]:
        raise ValueError(f"mode must be either 'complete' or 'partial', but got '{mode}'")

    df = pd.DataFrame(matched_df, copy=True)

    df = df[df[("left", "blink_type")] == mode]
    df = df[df[("right", "blink_type")] == mode]

    # drop all rows that are true in the "single" column
    if df[("left", "single")].dtype != bool:
        df[("left", "single")] = df[("left", "single")].astype(bool)
        df[("right", "single")] = df[("right", "single")].astype(bool)

    df = df[~df[("left", "single")]]
    df = df[~df[("right", "single")]]

    # compute the middle between left and right
    df["timestep"] = (df[("left", "apex_frame_og")] + df[("right", "apex_frame_og")]) / 2
    df["timestep_ms"] = df["timestep"] / fps * 1000  # convert to ms
    df["timestep_min"] = df["timestep_ms"] / 1000 / 60  # convert to min

    # compute the distance between left and right, and convert to ms
    df["distance"] = df[("left", "apex_frame_og")] - df[("right", "apex_frame_og")]
    df["distance_ms"] = df["distance"] / fps * 1000  # convert to ms
    df["distance_min"] = df["distance_ms"] / 1000 / 60  # convert to min

    dists_avg_roll = df["distance_ms"].rolling(rolling_mean).mean().fillna(method="bfill")
    dists_std_roll = df["distance_ms"].rolling(rolling_mean).std().fillna(method="bfill")

    # Start with a square Figure
    fig = plt.figure(figsize=(20, 6), dpi=300)
    gs = fig.add_gridspec(2, 2, width_ratios=(8, 1), height_ratios=(1, 4), left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.05)

    axis_main = fig.add_subplot(gs[1, 0])
    axis_main.scatter(df["timestep_min"], df["distance_ms"], c="k")

    (roll_avg,) = axis_main.plot(df["timestep_min"], dists_avg_roll, c="b")
    roll_std = axis_main.fill_between(df["timestep_min"], dists_avg_roll - dists_std_roll, dists_avg_roll + dists_std_roll, color="b", alpha=0.1, label=f"Rolling std [{rolling_mean}]")

    # draw the mean std as area around the mean
    axis_main.set_ylabel("Time Difference [ms] @ 240Hz")
    axis_main.set_xlabel("Time [min]")
    avg = df["distance_ms"].mean()
    std = df["distance_ms"].std()
    # get the extrem values
    outer = np.abs(df["distance_ms"]).max()

    # draw everything below 0 in red and everything above 0 in blue
    # ax.axhline(0, color="k", alpha=0.5)
    area_upper = axis_main.axhspan(0, outer, facecolor="r", alpha=0.1, hatch=".", edgecolor="r")
    area_lower = axis_main.axhspan(0, -outer, facecolor="b", alpha=0.1, hatch="x", edgecolor="b")

    axis_main.legend([roll_avg, roll_std, area_upper, area_lower], [f"Rolling Avg [{rolling_mean}]", f"Rolling Std [{rolling_mean}]", "Upper", "Lower"], loc="lower left")

    axis_main.text(axis_main.get_xlim()[0], avg + 1, f"{avg:.2f} ms", color="b")
    axis_main.axhline(avg, color="k", alpha=0.5)

    # # draw a line at the mean, and at the mean ± std, and at the mean ± 2*std, and at the mean ± 3*std
    axis_main.axhline(avg + std, color="k", alpha=0.4, linestyle="-")
    axis_main.axhline(avg - std, color="k", alpha=0.4, linestyle="-")
    axis_main.axhline(avg + std * 2, color="k", alpha=0.3, linestyle="--")
    axis_main.axhline(avg - std * 2, color="k", alpha=0.3, linestyle="--")
    axis_main.axhline(avg + std * 3, color="k", alpha=0.2, linestyle="-.")
    axis_main.axhline(avg - std * 3, color="k", alpha=0.2, linestyle="-.")

    ## Deviation from the mean
    # Add a new axis for the deviation from the mean
    axis_devi = axis_main.twinx()
    axis_devi.set_ylabel("Deviation from mean in [ms]")
    ylims = axis_main.get_ylim()
    axis_devi.set_ylim(ylims)
    axis_devi.set_yticks([-outer, avg - 3 * std, avg - 2 * std, avg - std, avg, avg + std, avg + 2 * std, avg + 3 * std, outer])
    axis_devi.set_yticklabels(["", "-3o", "-2o", "-o", "mean", "+o", "+2o", "+3o", ""])

    # group the dataframe by stds and count the number of blinks
    group_index = np.array([avg - outer, avg - 3 * std, avg - 2 * std, avg - std, avg, avg + std, avg + 2 * std, avg + 3 * std, avg + outer])
    # check that the group_index is monotonically increasing
    if np.all(np.diff(group_index) > 0):
        # Add a new axis for the histogram
        axis_hist = fig.add_subplot(gs[1, 1])
        groups = df.groupby(pd.cut(df["distance_ms"], group_index))
        groups_count = groups.count()
        # make the bins based on the stds
        # axis_hist.hist(df["distance_ms"], bins="sqrt", color="k", orientation="horizontal")
        bars = axis_hist.barh(group_index[:-1] + std / 2, groups_count["distance_ms"], color="k", height=std * 0.9)
        axis_hist.bar_label(bars, fmt="%d", label_type="center", color="w")
        # shift the ticks slightly higher to correct the bin centering
        axis_hist.yaxis.set_ticks_position("right")
        axis_hist.set_ylim(ylims)
        axis_hist.yaxis.set_visible(False)
        axis_hist.set_title("Histogram")

    # Add a new axis for the time histogram
    # 1. convert the frame_og to minutes based on the fps
    df[("left", "minute")] = df[("left", "apex_frame_og")] / fps / 60
    df[("right", "minute")] = df[("right", "apex_frame_og")] / fps / 60

    # 2. convert the minutes to datetime
    times_l = pd.to_datetime(df[("left", "minute")], unit="m", errors="ignore")
    times_r = pd.to_datetime(df[("right", "minute")], unit="m", errors="ignore")

    # 3. group the dataframe by the minutes
    group_l = df.groupby(times_l.dt.minute)
    group_r = df.groupby(times_r.dt.minute)

    # 4. create a summary dataframe
    summary_df = pd.DataFrame()
    summary_df["blinks_l"] = group_l.count()[("left", "minute")]
    summary_df["blinks_r"] = group_r.count()[("right", "minute")]
    summary_df["blinks"] = (summary_df["blinks_l"] + summary_df["blinks_r"]) / 2
    axis_time = fig.add_subplot(gs[0, 0])
    bars = axis_time.bar(summary_df.index + 0.5, summary_df["blinks"], color="k", width=0.9, align="center")
    axis_time.bar_label(bars, fmt="%d", label_type="center", color="w")

    axis_time.xaxis.set_visible(False)
    axis_time.set_ylabel("Blinks per\nminute")

    # get the maximal minute of the video based on the last row of the summary_df index
    max_minute = summary_df.index[-1] + 2
    axis_main.set_xticks(np.arange(0, max_minute, 1))

    # add a grid for the minutes
    axis_main.xaxis.grid(True, which="major")
    axis_main.set_xlim(-0.8)
    axis_time.set_xlim(axis_main.get_xlim())

    fig.suptitle(f"Time difference between left and right eye for {mode} matched blinks")

    # canvas = FigureCanvasAgg(fig)
    fig.canvas.draw()
    rgba_buf = fig.canvas.buffer_rgba()
    (w, h) = fig.canvas.get_width_height()
    rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h, w, 4))
    return rgba_arr
