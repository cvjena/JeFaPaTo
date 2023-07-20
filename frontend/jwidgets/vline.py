__all__ = ["JVLine"]

from PyQt6.QtWidgets import QFrame

class JVLine(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Sunken)