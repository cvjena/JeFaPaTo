__all__ = ["summarize", "visualize"]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def group_avg(group: pd.DataFrame, col: str, precision: int=1) -> pd.Series:
    mean = group.mean(numeric_only=True)[col].round(precision)
    return mean

def group_std(group: pd.DataFrame, col: str, precision: int = 1) -> pd.Series:
    std = group.std(numeric_only=True)[col].round(precision)
    return std
    
def calculate_statistics(summary_df, group, col, precision=1):
    summary_df[f"{col}_avg [#]"] = group_avg(group, col, precision)
    summary_df[f"{col}_std [#]"] = group_std(group, col, precision)
    
def summarize(
    matched_blinks: pd.DataFrame,
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

    Parameters:
        matched_blinks (pd.DataFrame): DataFrame containing the matched blinks data.
        fps (int): Frames per second. Default is 240.

    Returns:
        pd.DataFrame: DataFrame containing the summarized statistics for each minute.
    """
    df = pd.DataFrame(matched_blinks, copy=True)
    
    def _compute_statistics(df_i: pd.DataFrame, df_o: pd.DataFrame, side: str):
        df_i[(side,  "minute")] = df_i[(side,  "frame_og")]  / fps / 60
        times_l = pd.to_datetime(df_i[(side,  "minute")], unit='m', errors="ignore")
        group_l = df_i.groupby(times_l.dt.minute)
        df_o["blinks"] = group_l.count()[(side, "minute")]
        calculate_statistics(df_o, group_l, (side,  "width"))
        calculate_statistics(df_o, group_l, (side,  "height"), precision=2)
        
    summary_df = pd.DataFrame()
    _compute_statistics(df, summary_df, "left")
    _compute_statistics(df, summary_df, "right")
    summary_df.index.name = "minute"
    return summary_df

def visualize(
    matched_df: pd.DataFrame,
    fps: int = 240,
    rolling_mean: int = 5,
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
    df = pd.DataFrame(matched_df, copy=True)
    # drop all rows that are true in the "single" column
    if df[(("left", "single"))].dtype != bool:
        df[("left",  "single")] = df[("left",  "single")].astype(bool)
        df[("right", "single")] = df[("right", "single")].astype(bool)
    
    df = df[~df[("left", "single")]]
    df = df[~df[("right", "single")]]
    
    # compute the middle between left and right
    df["timestep"] = (df[("left", "frame_og")] + df[("right", "frame_og")] ) / 2
    df["timestep_ms"]  = df["timestep"] / fps * 1000   # convert to ms
    df["timestep_min"] = df["timestep_ms"] / 1000 / 60 # convert to min
    
    # compute the distance between left and right, and convert to ms
    df["distance"] = df[("left", "frame_og")] - df[("right", "frame_og")]
    df["distance_ms"]  = df["distance"] / fps * 1000   # convert to ms
    df["distance_min"] = df["distance_ms"] / 1000 / 60 # convert to min
    
    dists_avg_roll = df["distance_ms"].rolling(rolling_mean).mean().fillna(method="bfill")
    dists_std_roll = df["distance_ms"].rolling(rolling_mean).std().fillna(method="bfill")

    # Start with a square Figure
    fig = plt.figure(figsize=(20, 6), dpi=300)
    gs = fig.add_gridspec(2, 2,  width_ratios=(8, 1), height_ratios=(1, 4), left=0.1, right=0.9, bottom=0.1, top=0.9,wspace=0.1, hspace=0.05)

    axis_main = fig.add_subplot(gs[1, 0])
    axis_main.scatter(df["timestep_min"], df["distance_ms"], c="k")

    roll_avg, = axis_main.plot(df["timestep_min"], dists_avg_roll, c="b")
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
    area_upper = axis_main.axhspan(0,  outer, facecolor='r', alpha=0.1, hatch=".", edgecolor="r")
    area_lower = axis_main.axhspan(0, -outer, facecolor='b', alpha=0.1, hatch="x", edgecolor="b")

    axis_main.legend(
        [roll_avg, roll_std, area_upper, area_lower],
        [f"Rolling Avg [{rolling_mean}]", f"Rolling Std [{rolling_mean}]", "Upper", "Lower"],
        loc="lower left"
    )

    axis_main.text(axis_main.get_xlim()[0], avg+1, f"{avg:.2f} ms", color="b")
    axis_main.axhline(avg, color="k", alpha=0.5)
    
    # # draw a line at the mean, and at the mean ± std, and at the mean ± 2*std, and at the mean ± 3*std
    axis_main.axhline(avg + std,   color='k', alpha=0.4, linestyle="-")
    axis_main.axhline(avg - std,   color='k', alpha=0.4, linestyle="-")
    axis_main.axhline(avg + std*2, color='k', alpha=0.3, linestyle="--")
    axis_main.axhline(avg - std*2, color='k', alpha=0.3, linestyle="--")
    axis_main.axhline(avg + std*3, color='k', alpha=0.2, linestyle="-.")
    axis_main.axhline(avg - std*3, color='k', alpha=0.2, linestyle="-.")
    
    ## Deviation from the mean
    # Add a new axis for the deviation from the mean    
    axis_devi = axis_main.twinx()
    axis_devi.set_ylabel("Deviation from mean in [ms]")
    ylims = axis_main.get_ylim()
    axis_devi.set_ylim(ylims)
    axis_devi.set_yticks([-outer, avg-3*std, avg-2*std, avg-std, avg, avg+std, avg+2*std, avg+3*std, outer])
    axis_devi.set_yticklabels(["", "-3o", "-2o", "-o", "mean", "+o", "+2o", "+3o", ""])
    
    # Add a new axis for the histogram
    axis_hist = fig.add_subplot(gs[1, 1])
    
    # group the dataframe by stds and count the number of blinks
    group_index = np.array([avg-outer, avg-3*std, avg-2*std, avg-std, avg, avg+std, avg+2*std, avg+3*std, avg+outer])
    # check that the group_index is monotonically increasing
    if np.all(np.diff(group_index) > 0):
        groups = df.groupby(pd.cut(df["distance_ms"], group_index))
        groups_count = groups.count()
        # make the bins based on the stds
        # axis_hist.hist(df["distance_ms"], bins="sqrt", color="k", orientation="horizontal")
        bars = axis_hist.barh(group_index[:-1]+std/2, groups_count["distance_ms"], color="k", height=std*0.9)
        axis_hist.bar_label(bars, fmt="%d", label_type="center", color="w")
        # shift the ticks slightly higher to correct the bin centering
        axis_hist.yaxis.set_ticks_position("right")
        axis_hist.set_ylim(ylims)
        axis_hist.yaxis.set_visible(False)
        axis_hist.set_title("Histogram")
        
    # Add a new axis for the time histogram
    # 1. convert the frame_og to minutes based on the fps
    df[("left",  "minute")] = df[("left",  "frame_og")]  / fps / 60
    df[("right", "minute")] = df[("right", "frame_og")]  / fps / 60
    
    # 2. convert the minutes to datetime
    times_l = pd.to_datetime(df[("left",  "minute")], unit='m', errors="ignore")
    times_r = pd.to_datetime(df[("right", "minute")], unit='m', errors="ignore")

    # 3. group the dataframe by the minutes
    group_l = df.groupby(times_l.dt.minute)
    group_r = df.groupby(times_r.dt.minute)
    
    # 4. create a summary dataframe
    summary_df = pd.DataFrame()
    summary_df["blinks_l"] = group_l.count()[("left", "minute")]
    summary_df["blinks_r"] = group_r.count()[("right", "minute")]
    summary_df["blinks"] = (summary_df["blinks_l"] + summary_df["blinks_r"]) / 2
    axis_time = fig.add_subplot(gs[0, 0])
    bars = axis_time.bar(summary_df.index+0.5, summary_df["blinks"], color="k", width=0.9, align="center")
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

    # canvas = FigureCanvasAgg(fig)
    fig.canvas.draw()
    rgba_buf = fig.canvas.buffer_rgba()
    (w, h) = fig.canvas.get_width_height()
    rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h, w, 4))
    return rgba_arr
