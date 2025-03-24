__all__ = ["JDialogWarn", "JDialogRunning"]

from PyQt6 import QtWidgets, QtCore


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


class JDialogRunning(QtWidgets.QProgressDialog):
    """
    Custom dialog box for displaying the running status of a process.
    """

    def __init__(self, text="Running..."):
        super().__init__()
        self.setLabelText(text)
        self.setCancelButton(None)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowCloseButtonHint, False)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowMinimizeButtonHint, False)
