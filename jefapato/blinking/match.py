__all__ = ['match']

import numpy as np
import pandas as pd

def match(
    blinking_left: pd.DataFrame, 
    blinking_right: pd.DataFrame,
    tolerance: int = 80, # in frames
) -> pd.DataFrame:
    """
    Matches the left and right blinking dataframes by the frame index inside the threshold.

    However, we have to consider that perhaps for a blink in the left eye, there is no blink in the right eye, and vice versa.
    For that case such a blink is marked as single, and the other eye is marked as missing.
    All values for the missing eye are set to -1.
    """

    # delete column called "index"
    blinking_left.drop(columns=["index"], inplace=True)
    blinking_right.drop(columns=["index"], inplace=True)

    blinking_left["single"]  = False
    blinking_right["single"] = False

    blinking_left["frame_og"]  = blinking_left["frame"]
    blinking_right["frame_og"] = blinking_right["frame"]

    merge_lr = pd.merge_asof(blinking_left, blinking_right, on='frame', tolerance=tolerance, suffixes=('_left', '_right'), direction='nearest', allow_exact_matches=True)
    merge_rl = pd.merge_asof(blinking_right, blinking_left, on='frame', tolerance=tolerance, suffixes=('_right', '_left'), direction='nearest', allow_exact_matches=True)

    merged = pd.concat([merge_lr, merge_rl]).drop_duplicates(subset=['score_left']).sort_values(by=['frame']).reindex()
    merged = merged.drop(columns=['frame'])
    merged = merged.reset_index(drop=True)

    # iterate over the rows and check the single condition
    # if for a row the left eye is missing, then the right eye is single (set value to True)
    # if for a row the right eye is missing, then the left eye is single (set value to True)
    for index, row in merged.iterrows():
        if np.isnan(row['single_left']):
            merged.at[index, ('single_right')] = True
        if np.isnan(row['single_right']):
            merged.at[index, ('signle_left')] = True 

    # create a multiheader for left and right
    old_header = blinking_left.columns.delete(0)
    new_header = pd.MultiIndex.from_product([['left', 'right'], old_header])
    merged.columns = new_header

    return merged