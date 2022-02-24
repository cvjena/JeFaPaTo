__all__ = ["DEFAULTS"]

ear = {
    "smooth": True,
    "smooth_size": "91",
    "smooth_poly": "5",
    "min_dist": "80",
    "min_height": "0.1",
    "min_prominence": "0.05",
    "fps": "240",
    "min_width": "10",
    "max_width": "150",
    "threshold_l": "0.4",
    "threshold_r": "0.4",
    "draw_width_height": False,
}

landmarks = {
    "backend": "dlib",
    "backend_options": {
        "dlib": "dlib",
        "mediapipe": "mediapipe",
    },
    "features": ["EARFeature"],
    "feature_ear": True,
}

DEFAULTS = {
    "ear": ear,
    "landmarks": landmarks,
}
