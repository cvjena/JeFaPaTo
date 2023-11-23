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

class JBlinkingTable(QWidget):
    selection_changed = pyqtSignal(int)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.table_layout = QHBoxLayout()
        self.setLayout(self.table_layout)

        self.table_blinking = QTableView()
        self.table_layout.addWidget(self.table_blinking, stretch=1)
        # # set the model such that the whole list is selected when a row is selected
        self.table_blinking.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table_blinking.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.table_blinking.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self.table_blinking.horizontalHeader().setSectionsMovable(False)

        # set the model
        self.model_blinking_type = QStandardItemModel()
        self.table_blinking.setModel(self.model_blinking_type)
        
        self.table_blinking.selectionModel().selectionChanged.connect(
            lambda selected, _: self.selection_changed.emit(selected.indexes()[0].row())
        )
        
    def reset(self):
        self.model_blinking_type.clear()

    def get_annotations(self) -> pd.DataFrame:
        texts = []
        for i in range(self.model_blinking_type.rowCount()):
            model_idx = self.model_blinking_type.index(i, 0)
            widget: QComboBox = self.table_blinking.indexWidget(model_idx) # type: ignore
            texts.append(widget.currentText())
        
        annotations = pd.DataFrame()
        annotations["EyelidClosureType"] = texts

        return annotations

    def set_data(self, blinking_matched: pd.DataFrame):
        assert blinking_matched is not None and isinstance(blinking_matched, pd.DataFrame)
        self.reset()

        for _, row in blinking_matched.iterrows():
            self.model_blinking_type.appendRow([QStandardItem("")] + to_qt_row(row))

        # TODO perhaps we can atleast estimate which kind of blinking it is?
        for i in range(len(blinking_matched)):
            self.table_blinking.setIndexWidget(self.model_blinking_type.index(i, 0), create_blinking_combobox())

        self.table_blinking.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        # get the colums from the left and right eye and combine them
        
        header = [f"{self.split_and_capitalize(c[1])} [{c[0][0].upper()}]" for c in blinking_matched.columns]
        self.model_blinking_type.setHorizontalHeaderLabels(["Eyelid Closure Type"] + header)
        
        # resize the columns such that the headers are visible
        self.table_blinking.resizeColumnsToContents() 
    def split_and_capitalize(self, text: str) -> str:
        return "\n".join([word.capitalize() for word in text.split("_")])