__all__ = ["JBlinkingTable"]

import pandas as pd
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QStandardItem, QStandardItemModel
from PyQt6.QtWidgets import QHeaderView, QSplitter, QTableView


def to_qt_row(row: pd.Series) -> list:
    return [QStandardItem(str(row[c])) for c in row.index]


# TODO make table nicer.
#   - add better headers? perhaps alreayd in the dataframe?
#   - remove the splitter
#   - make the table stretch to the full width
#   - add another 3rd table with checkboxes for the user to select which rows to keep

class JBlinkingTable(QSplitter):
    selection_changed = pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setOrientation(Qt.Orientation.Horizontal)

        self.table_left_eye = QTableView()
        self.table_right_eye = QTableView()
        self.addWidget(self.table_left_eye)
        self.addWidget(self.table_right_eye)

        # set the model such that the whole list is selected when a row is selected
        self.table_left_eye.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table_right_eye.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        # set the model such that only one row can be selected at a time
        self.table_left_eye.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.table_right_eye.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        # set the model such that the user cannot edit the table
        self.table_left_eye.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self.table_right_eye.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        # set the model such that the user cannot move the columns
        self.table_left_eye.horizontalHeader().setSectionsMovable(False)
        self.table_right_eye.horizontalHeader().setSectionsMovable(False)

        self.model_left_eye = QStandardItemModel()
        self.model_right_eye = QStandardItemModel()

        self.table_left_eye.setModel(self.model_left_eye)
        self.table_right_eye.setModel(self.model_right_eye)
        
        # if the user clicks on a row, the other table should also select the same row
        self.table_left_eye.selectionModel().selectionChanged.connect(
            lambda selected, deselected: self.table_right_eye.selectRow(selected.indexes()[0].row())
        )
        self.table_right_eye.selectionModel().selectionChanged.connect(
            lambda selected, deselected: self.table_left_eye.selectRow(selected.indexes()[0].row())
        )

        self.table_left_eye.selectionModel().selectionChanged.connect(
            lambda selected, deselected: self.selection_changed.emit(selected.indexes()[0].row())
        )

    def reset(self):
        self.model_left_eye.clear()
        self.model_right_eye.clear()

    def set_data(self, blinking_matched: pd.DataFrame):
        assert blinking_matched is not None and isinstance(blinking_matched, pd.DataFrame)

        self.reset()

        data_left = blinking_matched["left"]
        data_right = blinking_matched["right"]

        for _, row in data_left.iterrows():
            self.model_left_eye.appendRow(to_qt_row(row))

        for _, row in data_right.iterrows():
            self.model_right_eye.appendRow(to_qt_row(row))

        self.table_left_eye.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_right_eye.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        self.model_left_eye.setHorizontalHeaderLabels(list(data_left.columns))
        self.model_right_eye.setHorizontalHeaderLabels(list(data_right.columns))