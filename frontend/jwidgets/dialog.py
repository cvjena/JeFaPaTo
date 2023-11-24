__all__ = ["JDialogWarn"]

from PyQt6 import QtWidgets

class JDialogWarn(QtWidgets.QMessageBox):
    """
    Custom dialog box for displaying warning messages.

    Args:
        title (str): The title of the warning dialog.
        text (str): The main text content of the warning dialog.
        information (str, optional): Additional information to display in the warning dialog.

    Attributes:
        None

    Methods:
        __init__: Initializes the JDialogWarn instance.

    """
    def __init__(
        self, 
        title: str, 
        text: str,
        information: str = "",
    ):
        super().__init__()
        self.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        self.setWindowTitle(title)
        self.setText(text)
        self.setInformativeText(information)
        self.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        self.exec()