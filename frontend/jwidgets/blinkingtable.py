__all__ = ["JBlinkingTable"]

import pandas as pd
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QStandardItem, QStandardItemModel, QColor
from PyQt6.QtWidgets import QHeaderView, QTableView, QWidget, QHBoxLayout, QComboBox

def to_qt_row(row: pd.Series) -> list:
    return [QStandardItem(str(row[c])) for c in row.index]

def create_blinking_combobox() -> QComboBox:
    combobox = QComboBox()
    # behing each item there should be a color in the background for easier selection
    combobox.addItem("No Eyelid Closure")
    combobox.addItem("Partial Eyelid Closure")
    combobox.addItem("Full Eyelid Closure")

    combobox.setItemData(2, QColor(Qt.GlobalColor.green), Qt.ItemDataRole.BackgroundRole)
    combobox.setItemData(1, QColor(Qt.GlobalColor.yellow), Qt.ItemDataRole.BackgroundRole)
    combobox.setItemData(0, QColor(Qt.GlobalColor.red), Qt.ItemDataRole.BackgroundRole)

    # set the combobox hover color to be the same as the background color
    combobox.setStyleSheet("QComboBox::drop-down { background-color: transparent; }")

    # if the user selects an item, the background color should change
    combobox.currentIndexChanged.connect(
        lambda index: combobox.setStyleSheet(f"background-color: {combobox.itemData(index, Qt.ItemDataRole.BackgroundRole).name()};")
    ) 
    combobox.setCurrentIndex(2)
    return combobox

# TODO make table nicer.
#   - add better headers? perhaps alreayd in the dataframe?
class JBlinkingTable(QWidget):
    selection_changed = pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.table_layout = QHBoxLayout()
        self.setLayout(self.table_layout)

        self.table_blinking_type = QTableView()
        self.table_left_eye = QTableView()
        self.table_right_eye = QTableView()

        self.table_layout.addWidget(self.table_blinking_type, stretch=1)
        self.table_layout.addWidget(self.table_left_eye, stretch=4)
        self.table_layout.addWidget(self.table_right_eye, stretch=4)

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

        # set the model
        self.model_blinking_type = QStandardItemModel()
        self.model_left_eye = QStandardItemModel()
        self.model_right_eye = QStandardItemModel()

        self.table_blinking_type.setModel(self.model_blinking_type)
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
        self.model_blinking_type.clear()
        self.model_left_eye.clear()
        self.model_right_eye.clear()

    def get_annotations(self) -> pd.DataFrame:
        texts = []
        for i in range(self.model_blinking_type.rowCount()):
            model_idx = self.model_blinking_type.index(i, 0)
            widget: QComboBox = self.table_blinking_type.indexWidget(model_idx) # type: ignore
            texts.append(widget.currentText())
        
        annotations = pd.DataFrame()
        annotations["EyelidClosureType"] = texts

        return annotations

    def set_data(self, blinking_matched: pd.DataFrame):
        assert blinking_matched is not None and isinstance(blinking_matched, pd.DataFrame)

        self.reset()

        data_left = blinking_matched["left"]
        data_right = blinking_matched["right"]

        for _, row in data_left.iterrows():
            self.model_left_eye.appendRow(to_qt_row(row))

        for _, row in data_right.iterrows():
            self.model_right_eye.appendRow(to_qt_row(row))
            self.model_blinking_type.appendRow(QStandardItem(""))

        # TODO perhaps we can atleast estimate which kind of blinking it is?
        for i in range(len(data_left)):
            self.table_blinking_type.setIndexWidget(self.model_blinking_type.index(i, 0), create_blinking_combobox())

        self.table_left_eye.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_right_eye.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_blinking_type.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        self.model_left_eye.setHorizontalHeaderLabels(list(data_left.columns))
        self.model_right_eye.setHorizontalHeaderLabels(list(data_right.columns))
        self.model_blinking_type.setHorizontalHeaderLabels(["Eyelid Closure Type"])