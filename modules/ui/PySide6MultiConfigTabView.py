from modules.ui.BaseMultiConfigTabView import BaseMultiConfigTabView
from modules.ui.MultiConfigTabController import MultiConfigTabController
from modules.util.ui import pyside6_components
from modules.util.ui.pyside6_util import QtABCMeta

from PySide6.QtWidgets import QWidget


class PySide6MultiConfigTabView(BaseMultiConfigTabView, QWidget, metaclass=QtABCMeta):
    def __init__(self, master, controller: MultiConfigTabController, ui_state):
        QWidget.__init__(self, master)
        BaseMultiConfigTabView.__init__(self, pyside6_components)

        self.master = master
        self.controller = controller
        self.ui_state = ui_state

        self.scroll_frame = None
        self.mode_frame = None
        self.values_frame = None
        self.footer_frame = None
        self.values_row = 0

        self.refresh_ui()

    def refresh_ui(self):
        if self.scroll_frame is not None:
            self.scroll_frame.hide()
            self.scroll_frame.setParent(None)
            self.scroll_frame.deleteLater()

        self._building = True
        try:
            self.scroll_frame = QWidget(self)
            pyside6_components._layout(self).addWidget(self.scroll_frame, 0, 0)
            layout = pyside6_components._layout(self.scroll_frame)
            layout.setContentsMargins(
                pyside6_components.PAD, pyside6_components.PAD,
                pyside6_components.PAD, pyside6_components.PAD,
            )
            layout.setColumnStretch(1, 1)
            layout.setColumnStretch(2, 2)

            self.build_header(self.scroll_frame, self.controller, self.ui_state)
            self.__build_mode_frame()
            self.__build_footer_frame()
            pyside6_components._pack_form(self.scroll_frame)
        finally:
            self._building = False

    def __build_mode_frame(self):
        self.mode_frame = QWidget(self.scroll_frame)
        pyside6_components._layout(self.scroll_frame).addWidget(self.mode_frame, 2, 0, 1, 3)
        layout = pyside6_components._layout(self.mode_frame)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 2)

        next_row = self.build_mode_content(self.mode_frame, self.controller, self.ui_state)
        self.__build_values_frame(next_row)
        pyside6_components._pack_form(self.mode_frame)

    def __build_values_frame(self, row: int):
        self.values_row = row
        self.values_frame = QWidget(self.mode_frame)
        pyside6_components._layout(self.mode_frame).addWidget(self.values_frame, row, 0, 1, 3)
        layout = pyside6_components._layout(self.values_frame)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 2)

        self.apply_repaired_sweep_values(self.controller, self.ui_state)
        self.build_value_inputs(self.values_frame, self.controller, self.ui_state)
        pyside6_components._pack_form(self.values_frame)

    def __build_footer_frame(self):
        self.footer_frame = QWidget(self.scroll_frame)
        pyside6_components._layout(self.scroll_frame).addWidget(self.footer_frame, 3, 0, 1, 3)
        layout = pyside6_components._layout(self.footer_frame)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 2)

        self.build_footer(self.footer_frame, self.controller, self.ui_state)
        pyside6_components._pack_form(self.footer_frame)

    # --- BaseMultiConfigTabView hooks ------------------------------------------------------------

    def refresh_mode_frame(self):
        if self.mode_frame is not None:
            self.mode_frame.hide()
            # detach before deleteLater: deletion happens when the event loop next runs, and until
            # then the old widgets would still be part of this tab
            self.mode_frame.setParent(None)
            self.mode_frame.deleteLater()
        self._building = True
        try:
            self.__build_mode_frame()
        finally:
            self._building = False

    def refresh_values_frame(self):
        if self.values_frame is not None:
            self.values_frame.hide()
            self.values_frame.setParent(None)
            self.values_frame.deleteLater()
        self._building = True
        try:
            self.__build_values_frame(self.values_row)
        finally:
            self._building = False
