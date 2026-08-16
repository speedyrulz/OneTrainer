from modules.ui.BaseMultiConfigTabView import BaseMultiConfigTabView
from modules.ui.MultiConfigTabController import MultiConfigTabController
from modules.util.ui import ctk_components

import customtkinter as ctk


class CtkMultiConfigTabView(BaseMultiConfigTabView):
    def __init__(self, master, controller: MultiConfigTabController, ui_state):
        BaseMultiConfigTabView.__init__(self, ctk_components)

        self.master = master
        self.controller = controller
        self.ui_state = ui_state

        self.scroll_frame = None
        self.mode_frame = None
        self.values_frame = None
        self.footer_frame = None

        master.grid_rowconfigure(0, weight=1)
        master.grid_columnconfigure(0, weight=1)

        self.refresh_ui()

    def refresh_ui(self):
        if self.scroll_frame:
            self.scroll_frame.destroy()

        self._building = True
        try:
            self.scroll_frame = ctk.CTkScrollableFrame(self.master, fg_color="transparent")
            self.scroll_frame.grid(row=0, column=0, sticky="nsew")
            self.scroll_frame.grid_columnconfigure(0, weight=0)
            self.scroll_frame.grid_columnconfigure(1, weight=1)
            self.scroll_frame.grid_columnconfigure(2, weight=2)

            self.build_header(self.scroll_frame, self.controller, self.ui_state)
            self.__build_mode_frame()
            self.__build_footer_frame()
        finally:
            self._building = False

    def __build_mode_frame(self):
        self.mode_frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        self.mode_frame.grid(row=2, column=0, columnspan=3, sticky="nsew")
        self.mode_frame.grid_columnconfigure(0, weight=0)
        self.mode_frame.grid_columnconfigure(1, weight=1)
        self.mode_frame.grid_columnconfigure(2, weight=2)

        next_row = self.build_mode_content(self.mode_frame, self.controller, self.ui_state)
        self.__build_values_frame(next_row)

    def __build_values_frame(self, row: int):
        self.values_row = row
        self.values_frame = ctk.CTkFrame(self.mode_frame, fg_color="transparent")
        self.values_frame.grid(row=row, column=0, columnspan=3, sticky="nsew")
        self.values_frame.grid_columnconfigure(0, weight=0)
        self.values_frame.grid_columnconfigure(1, weight=1)
        self.values_frame.grid_columnconfigure(2, weight=2)

        self.apply_repaired_sweep_values(self.controller, self.ui_state)
        self.build_value_inputs(self.values_frame, self.controller, self.ui_state)

    def __build_footer_frame(self):
        self.footer_frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        self.footer_frame.grid(row=3, column=0, columnspan=3, sticky="nsew")
        self.footer_frame.grid_columnconfigure(0, weight=0)
        self.footer_frame.grid_columnconfigure(1, weight=1)
        self.footer_frame.grid_columnconfigure(2, weight=2)

        self.build_footer(self.footer_frame, self.controller, self.ui_state)

    # --- BaseMultiConfigTabView hooks ------------------------------------------------------------

    def refresh_mode_frame(self):
        if self.mode_frame:
            self.mode_frame.destroy()
        self._building = True
        try:
            self.__build_mode_frame()
        finally:
            self._building = False

    def refresh_values_frame(self):
        if self.values_frame:
            self.values_frame.destroy()
        self._building = True
        try:
            self.__build_values_frame(self.values_row)
        finally:
            self._building = False
