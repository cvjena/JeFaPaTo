__all__ = ["summarize"]

import pandas as pd

def group_avg(group: pd.DataFrame, col: str, precision: int=1) -> pd.Series:
    mean = group.mean(numeric_only=True)[col].round(precision)
    return mean

def group_std(group: pd.DataFrame, col: str, precision: int = 1) -> pd.Series:
    std = group.std(numeric_only=True)[col].round(precision)
    return std
    
def summarize(
    merged_blinking: pd.DataFrame,
    fps: int = 240,
) -> pd.DataFrame:
    df = pd.DataFrame(merged_blinking, copy=True)
    # group the dataframe by the minutes
    # then count the number of blinks per minute
    
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

    summary_df["width_l_avg [#]"] = group_avg(group_l, ("left", "width"))
    summary_df["width_l_std [#]"] = group_std(group_l, ("left", "width"))
    
    summary_df["width_r_avg [#]"] = group_avg(group_r, ("right", "width"))
    summary_df["width_r_std [#]"] = group_std(group_r, ("right", "width"))
    
    summary_df["height_l_avg [#]"] = group_avg(group_l, ("left", "height"), precision=2)
    summary_df["height_l_std [#]"] = group_std(group_l, ("left", "height"), precision=2)
    
    summary_df["height_r_avg [#]"] = group_avg(group_r, ("right", "height"), precision=2)
    summary_df["height_r_std [#]"] = group_std(group_r, ("right", "height"), precision=2)
    
    summary_df.index.name = "minute"
    return summary_df