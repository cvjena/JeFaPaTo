__all__ = ["load_ear_score"]

from pathlib import Path
import numpy as np
import pandas as pd

def load_ear_score(
    dataframe: pd.DataFrame | Path,
    column_left: str = "EAR2D6_l",
    column_right: str = "EAR2D6_r",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load eye aspect ratio (EAR) scores from a pandas DataFrame or a CSV file.

    Args:
        dataframe (pd.DataFrame | Path): The input pandas DataFrame or path to the CSV file.
        column_left (str, optional): The column name for the left eye EAR scores. Defaults to "EAR2D6_l".
        column_right (str, optional): The column name for the right eye EAR scores. Defaults to "EAR2D6_r".

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the left and right eye EAR scores as numpy arrays.

    Raises:
        FileNotFoundError: If the input path does not exist or is not a file.
        TypeError: If the input is not a pandas DataFrame.
        ValueError: If the specified columns are not present in the DataFrame or if the lengths of the left and right eye scores are not the same.
    """
    
    if isinstance(dataframe, Path):
        if not dataframe.exists():
            raise FileNotFoundError(f"File {dataframe} does not exist.")
        if not dataframe.is_file():
            raise FileNotFoundError(f"Path {dataframe} is not a file.")
        dataframe = pd.read_csv(dataframe)
    
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Dataframe is not a pandas dataframe nor was a Path.")
    
    if column_left == column_right:
        raise ValueError("The column names for the left and right eye are the same.")
    
    if column_left not in dataframe.columns:
        raise ValueError(f"Column {column_left} is not in the dataframe.")
    
    if column_right not in dataframe.columns:
        raise ValueError(f"Column {column_right} is not in the dataframe.")
    
    left = dataframe[column_left].to_numpy()
    right = dataframe[column_right].to_numpy()
    return left, right