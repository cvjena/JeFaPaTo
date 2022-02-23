__all__ = ["smooth"]

import numpy as np
from scipy import signal


def smooth(time_series: np.ndarray, **kwargs) -> np.ndarray:
    kwargs["smooth_size"] = kwargs.get("smooth_size", 150)
    kwargs["smooth_poly"] = kwargs.get("smooth_poly", 4)

    return signal.savgol_filter(
        time_series,
        window_length=kwargs["smooth_size"],
        polyorder=kwargs["smooth_poly"],
    )
