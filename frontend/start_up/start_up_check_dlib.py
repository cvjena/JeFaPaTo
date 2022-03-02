import pathlib

from qtpy import QtWidgets

from .enum_state import StartUpState


def start_up_check_dlib(splash_screen: QtWidgets.QSplashScreen, **_) -> StartUpState:
    # check if dlib files are available
    # wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    # bunzip2 shape_predictor_68_face_landmarks.dat.bz2

    splash_screen.showMessage("Checking dlib files...")

    static_path = pathlib.Path(__file__).parent.parent.parent / "__static__"
    file_path = static_path / "shape_predictor_68_face_landmarks.dat"

    if file_path.exists() and file_path.is_file():
        splash_screen.showMessage("Dlib files found")
        return StartUpState.SUCCESS

    return download_files(file_path, splash_screen)


def download_files(
    file_path: pathlib.Path, splash_screen: QtWidgets.QSplashScreen
) -> StartUpState:
    import bz2

    import requests

    splash_screen.showMessage("Download dlib files...")
    file_path.parent.mkdir(parents=True, exist_ok=True)

    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

    try:
        res = requests.get(url)
        res.raise_for_status()
    except requests.exceptions.RequestException as e:
        splash_screen.showMessage(f"Error: {e}")
        return StartUpState.FAILURE

    splash_screen.showMessage("Writing dlib shape predictor...")

    try:
        file_path.write_bytes(bz2.decompress(res.content))
    except Exception as e:
        splash_screen.showMessage(f"Error: {e}")
        return StartUpState.FAILURE

    splash_screen.showMessage("Dlib files written")
    return StartUpState.SUCCESS
