__all__ = ["create_help_button", "to_float", "to_int", "F2S", "I2S"]

import qtawesome as qta
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QMessageBox


def create_help_button(tooltip: str, win=None) -> QtWidgets.QPushButton:
    """
    Create a help button with the given tooltip.

    Args:
        tooltip (str): The tooltip text to be displayed when hovering over the button.
        win (QtWidgets.QWidget, optional): The parent widget for the help button. Defaults to None.

    Returns:
        QtWidgets.QPushButton: The created help button.
    """
    help_btn = QtWidgets.QPushButton(qta.icon("fa5s.question-circle"), "")
    help_btn.setToolTip(tooltip)
    help_btn.clicked.connect(lambda: QMessageBox.information(win, "Help", tooltip))
    return help_btn


def to_float(value: str) -> float:
    """
    Converts a string value to a float.

    Args:
        value (str): The string value to be converted.

    Returns:
        float: The converted float value. If the conversion fails, returns 0.0.
    """
    try:
        return float(value)
    except ValueError:
        return 0.0


def to_int(value: str) -> int:
    """
    Converts a string value to an integer.

    Args:
        value (str): The string value to be converted.

    Returns:
        int: The converted integer value. If the conversion fails, returns 0.
    """
    try:
        return int(value)
    except ValueError:
        return 0


F2S = (lambda x: to_float(x), lambda x: str(x))
I2S = (lambda x: to_int(x), lambda x: str(x))
