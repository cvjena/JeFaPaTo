__all__ = ["JTableBlinking", "JTableSummary"]

import abc
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
    
    def reset(self):
        self.model.clear()
        
    def set_data(self, data: pd.DataFrame):
        """
        Needs to be implemented by the subclass. This method should set the data of the table.
        However, the subclass should call this method first to reset the table.
        """
        assert data is not None and isinstance(data, pd.DataFrame)
        self.reset()
        
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

    def get_annotations(self) -> pd.DataFrame:
        texts = []
        for i in range(self.model.rowCount()):
            model_idx = self.model.index(i, 0)
            widget: QComboBox = self.view.indexWidget(model_idx) # type: ignore
            texts.append(widget.currentText())
        
        annotations = pd.DataFrame()
        annotations["EyelidClosureType"] = texts

        return annotations

    def set_data(self, blinking_matched: pd.DataFrame):
        super().set_data(blinking_matched)

        for _, row in blinking_matched.iterrows():
            self.model.appendRow([QStandardItem("")] + to_qt_row(row))

        # TODO perhaps we can atleast estimate which kind of blinking it is?
        for i in range(len(blinking_matched)):
            self.view.setIndexWidget(self.model.index(i, 0), create_blinking_combobox())

        self.view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.model.setHorizontalHeaderLabels(["Eyelid Closure Type"] + self.__parse_header(blinking_matched.columns))
        
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