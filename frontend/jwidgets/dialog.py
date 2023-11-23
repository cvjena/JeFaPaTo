__all__ = ["JDialogWarn"]

from PyQt6 import QtWidgets

class JDialogWarn(QtWidgets.QMessageBox):
    def __init__(
        self, 
        title: str, 
        text:str,
        information: str = "",
    ):
        super().__init__()
        self.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        self.setWindowTitle(title)
        self.setText(text)
        self.setInformativeText(information)
        self.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        self.exec()