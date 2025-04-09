__all__ = ["load_ear_score", "save_results"]

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


def save_results(
    filename: Path | str,
    blinking_l: pd.DataFrame | None = None,
    blinking_r: pd.DataFrame | None = None,
    blinking_matched: pd.DataFrame | None = None,
    blinking_summary: pd.DataFrame | None = None,
    format: str = "excel",
    exists_ok: bool = False,
):
    if isinstance(filename, str):
        filename = Path(filename)

    if format not in ["excel", "csv"]:
        raise ValueError(f"Invalid format {format}.")

    if blinking_l is None and blinking_r is None and blinking_matched is None and blinking_summary is None:
        raise RuntimeError("No data to save.")

    if format == "excel":
        excel_file = filename.parent / (filename.stem + "_blinking.xlsx")

        if excel_file.exists() and not exists_ok:
            raise FileExistsError(f"File {excel_file} already exists.")

        with pd.ExcelWriter(excel_file) as writer:
            if blinking_l is not None:
                blinking_l.to_excel(writer, sheet_name="blinking left eye")

            if blinking_r is not None:
                blinking_r.to_excel(writer, sheet_name="blinking right eye")

            if blinking_matched is not None:
                blinking_matched[("left", "single")] = blinking_matched[("left", "single")].astype(str)
                blinking_matched[("right", "single")] = blinking_matched[("right", "single")].astype(str)
                blinking_matched.to_excel(writer, sheet_name="Matched")

            if blinking_summary is not None:
                blinking_summary.to_excel(writer, sheet_name="Summary")
        return
    elif format == "csv":
        csv_file = filename.parent / (filename.stem + "_left.csv")
        if blinking_l is not None and (not csv_file.exists() or (csv_file.exists() and exists_ok)):
            blinking_l.to_csv(csv_file)

        csv_file = filename.parent / (filename.stem + "_right.csv")
        if blinking_r is not None and (not csv_file.exists() or (csv_file.exists() and exists_ok)):
            blinking_r.to_csv(csv_file)

        csv_file = filename.parent / (filename.stem + "_matched.csv")
        if blinking_matched is not None and (not csv_file.exists() or (csv_file.exists() and exists_ok)):
            blinking_matched.to_csv(csv_file)

        csv_file = filename.parent / (filename.stem + "_summary.csv")
        if blinking_summary is not None and (not csv_file.exists() or (csv_file.exists() and exists_ok)):
            blinking_summary.to_csv(csv_file)

        return
    else:
        raise ValueError(f"Invalid format {format}.")
