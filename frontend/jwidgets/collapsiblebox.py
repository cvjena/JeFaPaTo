__all__ = ["JCollapsibleBox"]

from PyQt6.QtCore import QAbstractAnimation, QByteArray, QParallelAnimationGroup, QPropertyAnimation, Qt, QVariantAnimation, pyqtSlot
from PyQt6.QtWidgets import QFrame, QScrollArea, QSizePolicy, QToolButton, QVBoxLayout, QWidget


class JCollapsibleBox(QWidget):
    # https://stackoverflow.com/questions/52615115/how-to-create-collapsible-box-in-pyqt/52617714#52617714
    def __init__(self, title="", parent=None):
        super().__init__(parent)

        self.toggle_button = QToolButton()
        self.toggle_button.setCheckable(True)
        self.toggle_button.setText(title)
        self.toggle_button.setChecked(False)
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle_button.pressed.connect(self.on_pressed)

        self.toggle_animation = QParallelAnimationGroup(self)

        self.content_area = QScrollArea()
        self.content_area.setMaximumHeight(0)
        self.content_area.setMinimumHeight(0)
        self.content_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.content_area.setFrameShape(QFrame.Shape.NoFrame)

        lay = QVBoxLayout(self)
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toggle_button)
        lay.addWidget(self.content_area)

        self.toggle_animation.addAnimation(QPropertyAnimation(self,              QByteArray("minimumHeight".encode()))) # type: ignore
        self.toggle_animation.addAnimation(QPropertyAnimation(self,              QByteArray("maximumHeight".encode()))) # type: ignore
        self.toggle_animation.addAnimation(QPropertyAnimation(self.content_area, QByteArray("maximumHeight".encode()))) # type: ignore

    @pyqtSlot()
    def on_pressed(self):
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(Qt.ArrowType.DownArrow if not checked else Qt.ArrowType.RightArrow)
        self.toggle_animation.setDirection(QAbstractAnimation.Direction.Forward if not checked else QAbstractAnimation.Direction.Backward)
        self.toggle_animation.start()

    def setContentLayout(self, layout):
        lay = self.content_area.layout()
        del lay
        self.content_area.setLayout(layout)
        collapsed_height = self.sizeHint().height() - self.content_area.maximumHeight()
        content_height = layout.sizeHint().height()
        for i in range(self.toggle_animation.animationCount()):
            animation = self.toggle_animation.animationAt(i) # type: QVariantAnimation # type: ignore
            animation.setDuration(500)
            animation.setStartValue(collapsed_height)
            animation.setEndValue(collapsed_height + content_height)

        content_animation = self.toggle_animation.animationAt(self.toggle_animation.animationCount() - 1) # type: QVariantAnimation # type: ignore
        content_animation.setDuration(500)
        content_animation.setStartValue(0)
        content_animation.setEndValue(content_height)