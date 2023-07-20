__all__ = ["JBlinkingTable"]

from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtWidgets import QTableView, QSplitter, QHeaderView
from PyQt6.QtCore import Qt
import pandas as pd

def to_qt_row(row: pd.Series) -> list:
    return [QStandardItem(str(row[c])) for c in row.index]


class JBlinkingTable(QSplitter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setOrientation(Qt.Orientation.Horizontal)

        self.table_left_eye = QTableView()
        self.table_right_eye = QTableView()

        self.model_left_eye = QStandardItemModel()
        self.model_right_eye = QStandardItemModel()

        self.table_left_eye.setModel(self.model_left_eye)
        self.table_right_eye.setModel(self.model_right_eye)

        self.addWidget(self.table_left_eye)
        self.addWidget(self.table_right_eye)


    def reset(self):
        self.model_left_eye.clear()
        self.model_right_eye.clear()


    def set_data(self, data_left: pd.DataFrame, data_right: pd.DataFrame):
        assert data_left is not None and isinstance(data_left, pd.DataFrame)
        assert data_right is not None and isinstance(data_right, pd.DataFrame)

        self.reset()

        for _, row in data_left.iterrows():
            self.model_left_eye.appendRow(to_qt_row(row))

        for _, row in data_right.iterrows():
            self.model_right_eye.appendRow(to_qt_row(row))

        self.table_left_eye.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_right_eye.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        self.model_left_eye.setHorizontalHeaderLabels(list(data_left.columns))
        self.model_right_eye.setHorizontalHeaderLabels(list(data_right.columns))