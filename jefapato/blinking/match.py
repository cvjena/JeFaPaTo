__all__ = ['match']

import numpy as np
import pandas as pd

def match(
    blinking_l: pd.DataFrame, 
    blinking_r: pd.DataFrame,
    tolerance: int = 30, # in frames
) -> pd.DataFrame:
    """
    Matches the left and right blinking dataframes by the frame index inside the threshold.

    However, we have to consider that perhaps for a blink in the left eye, there is no blink in the right eye, and vice versa.
    For that case such a blink is marked as single, and the other eye is marked as missing.
    All values for the missing eye are set to np.nan.
    
    Parameters
    ----------
    
    blinking_l : pd.DataFrame
        Blinking dataframe for the left eye. Extracted from the left eye EAR score, requires the column "frame".
    blinking_r : pd.DataFrame
        Blinking dataframe for the right eye. Extracted from the right eye EAR score, requires the column "frame".
    tolerance : int
        The tolerance in frames for the matching of the left and right eye, default is 30 frames.
    
    Returns
    -------
    
    pd.DataFrame
        A dataframe with the matched left and right eye, with same columns as the input dataframes but with _left and _right suffixes.
    
    Raises
    ------
    
    ValueError
        - If the dataframe for the left eye is None.
        - If the dataframe for the right eye is None.
        - If the dataframe for the left eye is not a pandas dataframe.
        - If the dataframe for the right eye is not a pandas dataframe.
        - If the dataframe for the left eye is empty.
        - If the dataframe for the right eye is empty.
        - If the tolerance is negative.
        - If the dataframe for the left eye does not have a column called 'frame'.
        - If the dataframe for the right eye does not have a column called 'frame'.
    
    """
    if blinking_l is None:
        raise ValueError("Dataframe for left eye is None.")
    if blinking_r is None:
        raise ValueError("Dataframe for right eye is None.")
    if not isinstance(blinking_l, pd.DataFrame):
        raise TypeError("Dataframe for left eye is not a pandas dataframe.")
    if not isinstance(blinking_r, pd.DataFrame):
        raise TypeError("Dataframe for right eye is not a pandas dataframe.")
    
    if blinking_l.empty:
        raise ValueError("Dataframe for left eye is empty.")
    if blinking_r.empty:
        raise ValueError("Dataframe for right eye is empty.")
   
    if tolerance < 0:
        raise ValueError("Tolerance is negative.") 
    
    # delete column called "index"
    if "index" in blinking_l.columns:
        blinking_l.drop(columns=["index"], inplace=True)
    if "index" in blinking_r.columns:
        blinking_r.drop(columns=["index"], inplace=True)
        
    if "frame" not in blinking_l.columns:
        raise ValueError("Dataframe for left eye does not have a column called 'frame'.")
    if "frame" not in blinking_r.columns:
        raise ValueError("Dataframe for right eye does not have a column called 'frame'.")
        
    blinking_l["single"] = False
    blinking_r["single"] = False
    
    blinking_l["frame_og"] = blinking_l["frame"]
    blinking_r["frame_og"] = blinking_r["frame"]

    merge_lr = pd.merge_asof(blinking_l, blinking_r, on='frame', tolerance=tolerance, suffixes=('_left', '_right'), direction='nearest', allow_exact_matches=True)
    merge_rl = pd.merge_asof(blinking_r, blinking_l, on='frame', tolerance=tolerance, suffixes=('_right', '_left'), direction='nearest', allow_exact_matches=True)

    merged = pd.concat([merge_lr, merge_rl])
    merged = merged.drop_duplicates(subset=["frame_og_left", "frame_og_right"]).sort_values(by=['frame']).reindex()
    merged = merged.drop(columns=['frame'])
    merged = merged.reset_index(drop=True)
    
    # check that all frames from blinking_l are in mergede["frame_og_left"]
    assert blinking_l["frame_og"].isin(merged["frame_og_left"]).all()
    # check that all frames from blinking_r are in mergede["frame_og_right"]
    assert blinking_r["frame_og"].isin(merged["frame_og_right"]).all()

    # iterate over the rows and check the single condition
    # if for a row the left eye is missing, then the right eye is single (set value to True)
    # if for a row the right eye is missing, then the left eye is single (set value to True)
    for index, row in merged.iterrows():
        if np.isnan(row['single_left']):
            merged.at[index, ('single_right')] = True
        if np.isnan(row['single_right']):
            merged.at[index, ('single_left')] = True 

    # create a multiheader for left and right
    old_header = blinking_l.columns.delete(0)
    new_header = pd.MultiIndex.from_product([['left', 'right'], old_header])
    merged.columns = new_header

    return merged