__all__ = ["JTableBlinking", "JTableSummary"]

import abc
from typing import Callable
import pandas as pd
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QStandardItem, QStandardItemModel, QColor
from PyQt6.QtWidgets import QHeaderView, QTableView, QWidget, QHBoxLayout, QComboBox

class JNumberItem(QStandardItem):
    def __init__(self, text: str):
        super().__init__(text)
        self.setTextAlignment(Qt.AlignmentFlag.AlignRight)
        self.setEditable(False)

def to_qt_row(row: pd.Series) -> list:
    return [JNumberItem(str(row[c])) for c in row.index]

def create_blinking_combobox(row_id: int, mode: str, blink_type: str | None = None, connection: Callable | None = None) -> QComboBox:
    combobox = QComboBox()
    
    combobox.addItem("None")
    combobox.addItem("Partial")
    combobox.addItem("Complete")

    combobox.setItemData(0, QColor(Qt.GlobalColor.white),  Qt.ItemDataRole.BackgroundRole)
    combobox.setItemData(1, QColor(Qt.GlobalColor.yellow), Qt.ItemDataRole.BackgroundRole)
    combobox.setItemData(2, QColor(Qt.GlobalColor.green),  Qt.ItemDataRole.BackgroundRole)

    # set the combobox hover color to be the same as the background color
    combobox.setStyleSheet("QComboBox::drop-down { background-color: transparent; }")

    # if the user selects an item, the background color should change
    combobox.currentIndexChanged.connect(
        lambda index: combobox.setStyleSheet(f"background-color: {combobox.itemData(index, Qt.ItemDataRole.BackgroundRole).name()};")
    )
  
    blink_type = blink_type if blink_type is not None else "none"
    if blink_type not in ["none", "partial", "complete"]:
        blink_type = "none"
    combobox.setCurrentText(blink_type.capitalize())
    
    # do the connection afterwords, because the combobox needs to be initialized first
    if connection is not None:
        combobox.currentIndexChanged.connect(lambda: connection(row_id, mode))
    
    return combobox
  
def split_and_capitalize(text: str) -> str:
    words = [word.capitalize() for word in text.split("_")]
    return " ".join(words)
    
class JTable(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # set the model such that the whole list is selected when a row is selected
        self.view = QTableView()
        self.view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.view.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.view.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self.view.horizontalHeader().setSectionsMovable(False)
    
        # alternate row colors
        self.view.setAlternatingRowColors(True)

        # set the model
        self.model = QStandardItemModel()
        self.view.setModel(self.model)
        
        self.setLayout(QHBoxLayout())
        self.layout().addWidget(self.view, stretch=1)
        
        self.data = None
    
    def reset(self):
        self.model.clear()
        
    def set_data(self, data: pd.DataFrame):
        """
        Needs to be implemented by the subclass. This method should set the data of the table.
        However, the subclass should call this method first to reset the table.
        """
        assert data is not None and isinstance(data, pd.DataFrame)
        self.reset()
        self.data = data
        
    @abc.abstractmethod
    def __parse_header(self, header: list) -> list:
        """
        Needs to be implemented by the subclass. This method should parse the header of the table.
        """


class JTableBlinking(JTable):
    selection_changed = pyqtSignal(int)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.view.selectionModel().selectionChanged.connect(
            lambda selected, _: self.selection_changed.emit(selected.indexes()[0].row())
        )

    def set_annotations(self, row_id: int, mode: str):
        if self.data is None:
            raise ValueError("No data set.")
        
        col_id = self.col_blink_type_l if mode == "left" else self.col_blink_type_r
        model_idx = self.model.index(row_id, col_id)
        widget: QComboBox = self.view.indexWidget(model_idx) # type: ignore

        type = widget.currentText().lower()
        print(f"Setting {row_id}, {mode} to {type}")
        # the data should be a reference to the origial data frame (and not a copy)
        self.data.loc[row_id, (mode, "blink_type")] = type

    def set_data(self, blinking_matched: pd.DataFrame):
        super().set_data(blinking_matched)
        for _, row in blinking_matched.iterrows():
            self.model.appendRow(to_qt_row(row))
            
        # get col_index of "blinking_type"
        self.col_blink_type_l = blinking_matched.columns.get_loc(("left", "blink_type"))
        self.col_blink_type_r = blinking_matched.columns.get_loc(("right", "blink_type"))
        
        for i in range(len(blinking_matched)):
            self.view.setIndexWidget(self.model.index(i, self.col_blink_type_l), create_blinking_combobox(i, "left",  blink_type=blinking_matched.iloc[i, self.col_blink_type_l], connection=self.set_annotations))
            self.view.setIndexWidget(self.model.index(i, self.col_blink_type_r), create_blinking_combobox(i, "right", blink_type=blinking_matched.iloc[i, self.col_blink_type_r], connection=self.set_annotations))

        self.view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.model.setHorizontalHeaderLabels(self.__parse_header(blinking_matched.columns))
        
        # resize the columns such that the headers are visible
        self.view.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignTop)
        self.view.resizeColumnsToContents() 
        self.view.resizeRowsToContents()
        self.view.horizontalHeader().setMinimumHeight(45)

    def __parse_header(self, header: list) -> list:
        return [f"[{c[0][0].upper()}]\n {split_and_capitalize(c[1])}" for c in header]
    
class JTableSummary(JTable):
    def set_data(self, blinking_summary: pd.DataFrame):
        super().set_data(blinking_summary)

        for _, row in blinking_summary.iterrows():
            self.model.appendRow(to_qt_row(row))

        self.view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.model.setHorizontalHeaderLabels(self.__parse_header(blinking_summary.columns))
        
        # resize the columns such that the headers are visible
        self.view.resizeColumnsToContents() 
        self.view.resizeRowsToContents()
        
        self.view.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignTop)
        
    def __parse_header(self, header: list) -> list:
        return [f"[{c[0][0].upper()}]\n {split_and_capitalize(c[1])}" for c in header]