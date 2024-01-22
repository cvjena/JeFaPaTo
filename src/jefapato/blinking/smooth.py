__all__ = ["smooth"]

import numpy as np
from scipy import signal


def smooth(time_series: np.ndarray, smooth_size:int=91, smooth_poly:int=4) -> np.ndarray:
    return signal.savgol_filter(
        time_series,
        window_length=smooth_size,
        polyorder=smooth_poly,
    )
