from PyQt5.QtWidgets import QHBoxLayout, QWidget


class WidgetEmotionRecognition(QWidget):
    def __init__(self):
        super().__init__()

        layout = QHBoxLayout()
        # layout.addWidget(QLabel("subjects"))
        # layout.addWidget(QCheckBox("Physics"))
        # layout.addWidget(QCheckBox("Maths"))

        self.setLayout(layout)
