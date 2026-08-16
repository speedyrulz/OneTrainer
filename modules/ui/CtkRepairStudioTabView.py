"""The Repair Studio tab, CustomTkinter.

Unlike the other tabs this one does not go through the shared `components` abstraction: those
components bind to a `ui_state` variable backed by a TrainConfig field, and none of this tab's state
lives in the config — it belongs to whichever LoRA happens to be open. So the widgets are built
directly, and everything that decides *what* to build lives in RepairStudioTabController, which both
frameworks share and the tests drive without a display.
"""

import os
from tkinter import filedialog

from modules.ui.RepairStudioTabController import RepairStudioTabController
from modules.util.ui.ToolTip import ToolTip

import customtkinter as ctk

SLIDER_STEPS = 600  # 0.01 across the -3..3 range

QUICK_SETS = [
    ("0", "zero", "Take this block out entirely — its keys are dropped from the saved file."),
    ("1", "full", "Back to the strength it was trained at."),
    ("±", "invert", "Flip the sign, so the block subtracts what it used to add."),
    ("⚖", "balance", "Hold primary + donor at 1.0 for this block, to cross-fade the two LoRAs."),
]


class CtkRepairStudioTabView:
    def __init__(self, master, controller: RepairStudioTabController):
        self.master = master
        self.controller = controller
        self.rows: dict[str, dict] = {}

        master.grid_rowconfigure(1, weight=1)
        master.grid_columnconfigure(0, weight=1)

        self.__build_header()

        self.scroll_frame = ctk.CTkScrollableFrame(master, fg_color="transparent")
        self.scroll_frame.grid(row=1, column=0, sticky="nsew", padx=6, pady=4)
        self.scroll_frame.grid_columnconfigure(0, weight=1)

        self.__build_footer()
        self.refresh()

    # --- chrome --------------------------------------------------------------------------------

    def __build_header(self):
        header = ctk.CTkFrame(self.master)
        header.grid(row=0, column=0, sticky="new", padx=6, pady=(6, 0))
        header.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(header, text="LoRA", width=90, anchor="w").grid(row=0, column=0, padx=6, pady=4)
        self.primary_entry = ctk.CTkEntry(header)
        self.primary_entry.grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        ctk.CTkButton(header, text="Browse", width=80, command=self.__browse_primary) \
            .grid(row=0, column=2, padx=4, pady=4)

        self.primary_label = ctk.CTkLabel(header, text="", anchor="w", justify="left")
        self.primary_label.grid(row=1, column=1, columnspan=2, sticky="ew", padx=4)

        ctk.CTkLabel(header, text="Donor", width=90, anchor="w").grid(row=2, column=0, padx=6, pady=4)
        self.donor_entry = ctk.CTkEntry(header)
        self.donor_entry.grid(row=2, column=1, sticky="ew", padx=4, pady=4)

        donor_buttons = ctk.CTkFrame(header, fg_color="transparent")
        donor_buttons.grid(row=2, column=2, padx=4, pady=4)
        ctk.CTkButton(donor_buttons, text="Browse", width=80, command=self.__browse_donor) \
            .grid(row=0, column=0)
        ctk.CTkButton(donor_buttons, text="Clear", width=60, command=self.__clear_donor) \
            .grid(row=0, column=1, padx=(4, 0))

        self.donor_label = ctk.CTkLabel(header, text="", anchor="w", justify="left")
        self.donor_label.grid(row=3, column=1, columnspan=2, sticky="ew", padx=4, pady=(0, 4))

        tools = ctk.CTkFrame(header, fg_color="transparent")
        tools.grid(row=4, column=0, columnspan=3, sticky="ew", padx=4, pady=(0, 6))

        ctk.CTkLabel(tools, text="All blocks:").grid(row=0, column=0, padx=(2, 6))
        for index, (text, action, tooltip) in enumerate(QUICK_SETS):
            button = ctk.CTkButton(
                tools, text=text, width=34,
                command=lambda a=action: self.__quick_set_all(a),
            )
            button.grid(row=0, column=1 + index, padx=2)
            ToolTip(button, f"Every block: {tooltip}")

        ctk.CTkButton(tools, text="Load preset", width=100, command=self.__load_preset) \
            .grid(row=0, column=6, padx=(16, 2))
        ctk.CTkButton(tools, text="Save preset", width=100, command=self.__save_preset) \
            .grid(row=0, column=7, padx=2)

    def __build_footer(self):
        footer = ctk.CTkFrame(self.master)
        footer.grid(row=2, column=0, sticky="sew", padx=6, pady=(0, 6))
        footer.grid_columnconfigure(1, weight=1)

        self.summary_label = ctk.CTkLabel(footer, text="", anchor="w", justify="left")
        self.summary_label.grid(row=0, column=0, columnspan=3, sticky="ew", padx=6, pady=(6, 2))

        ctk.CTkLabel(footer, text="Save as", width=90, anchor="w") \
            .grid(row=1, column=0, padx=6, pady=4)
        self.output_entry = ctk.CTkEntry(footer)
        self.output_entry.grid(row=1, column=1, sticky="ew", padx=4, pady=4)
        save_button = ctk.CTkButton(footer, text="Save baked LoRA", width=150, command=self.__save)
        save_button.grid(row=1, column=2, padx=4, pady=4)
        ToolTip(save_button, "Writes a normal .safetensors with the slider settings baked in. It "
                             "loads at strength 1.0 in ComfyUI and anywhere else — nothing has to "
                             "know it was edited.", wide=True)

        self.status_label = ctk.CTkLabel(footer, text="", anchor="w", justify="left")
        self.status_label.grid(row=2, column=0, columnspan=3, sticky="ew", padx=6, pady=(0, 6))

    # --- the sliders ----------------------------------------------------------------------------

    def refresh(self):
        """Rebuild the slider list. Called when the LoRA changes, not when a slider moves."""
        for child in self.scroll_frame.winfo_children():
            child.destroy()
        self.rows.clear()

        self.primary_label.configure(text=self.controller.primary_summary())
        self.donor_label.configure(text=self.controller.donor_summary())

        row = 0
        for heading, blocks in self.controller.grouped_blocks():
            ctk.CTkLabel(self.scroll_frame, text=heading, anchor="w",
                         font=ctk.CTkFont(weight="bold")) \
                .grid(row=row, column=0, sticky="ew", padx=6, pady=(10, 2))
            row += 1
            for block in blocks:
                self.__build_row(row, block)
                row += 1

        if not self.controller.blocks():
            ctk.CTkLabel(self.scroll_frame, text="No LoRA open.", anchor="w") \
                .grid(row=0, column=0, sticky="ew", padx=6, pady=10)

        self.__update_summary()

    def __build_row(self, row: int, block):
        low, high = self.controller.slider_range()
        frame = ctk.CTkFrame(self.scroll_frame)
        frame.grid(row=row, column=0, sticky="ew", padx=4, pady=2)
        frame.grid_columnconfigure(1, weight=1)

        count = self.controller.module_count(block.id)
        ctk.CTkLabel(frame, text=f"{block.label}  ({count})", width=170, anchor="w") \
            .grid(row=0, column=0, padx=(8, 4), pady=4)

        state = self.controller.block_state(block.id)
        widgets = {}

        slider = ctk.CTkSlider(frame, from_=low, to=high, number_of_steps=SLIDER_STEPS)
        slider.set(state.primary_strength)
        slider.configure(command=lambda v, b=block.id: self.__on_slide(b, v, donor=False))
        slider.grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        widgets["primary_slider"] = slider

        value = ctk.CTkLabel(frame, text=format_strength(state.primary_strength), width=52)
        value.grid(row=0, column=2, padx=2)
        widgets["primary_value"] = value

        buttons = ctk.CTkFrame(frame, fg_color="transparent")
        buttons.grid(row=0, column=3, padx=(4, 8))
        for index, (text, action, tooltip) in enumerate(QUICK_SETS):
            button = ctk.CTkButton(
                buttons, text=text, width=32,
                command=lambda a=action, b=block.id: self.__quick_set(b, a, donor=False),
            )
            button.grid(row=0, column=index, padx=1)
            ToolTip(button, tooltip)

        if self.controller.has_donor:
            ctk.CTkLabel(frame, text="donor", width=170, anchor="e") \
                .grid(row=1, column=0, padx=(8, 4), pady=(0, 4))

            donor_slider = ctk.CTkSlider(frame, from_=low, to=high, number_of_steps=SLIDER_STEPS,
                                         button_color="#fd7e14")
            donor_slider.set(state.donor_strength)
            donor_slider.configure(command=lambda v, b=block.id: self.__on_slide(b, v, donor=True))
            donor_slider.grid(row=1, column=1, sticky="ew", padx=4, pady=(0, 4))
            widgets["donor_slider"] = donor_slider

            donor_value = ctk.CTkLabel(frame, text=format_strength(state.donor_strength), width=52)
            donor_value.grid(row=1, column=2, padx=2)
            widgets["donor_value"] = donor_value

            donor_buttons = ctk.CTkFrame(frame, fg_color="transparent")
            donor_buttons.grid(row=1, column=3, padx=(4, 8), pady=(0, 4))
            for index, (text, action, _tip) in enumerate(QUICK_SETS):
                ctk.CTkButton(
                    donor_buttons, text=text, width=32,
                    command=lambda a=action, b=block.id: self.__quick_set(b, a, donor=True),
                ).grid(row=0, column=index, padx=1)

        self.rows[block.id] = widgets

    # --- reacting -------------------------------------------------------------------------------

    def __on_slide(self, block_id: str, value: float, *, donor: bool):
        self.controller.set_strength(block_id, float(value), donor=donor)
        self.__sync_row(block_id)
        self.__update_summary()

    def __quick_set(self, block_id: str, action: str, *, donor: bool):
        self.controller.apply(action, block_id, donor=donor)
        # balance moves the *other* side, so the whole row is resynced either way
        self.__sync_row(block_id)
        self.__update_summary()

    def __quick_set_all(self, action: str):
        self.controller.apply(action)
        for block_id in self.rows:
            self.__sync_row(block_id)
        self.__update_summary()

    def __sync_row(self, block_id: str):
        """Push the state back into the widgets, without rebuilding them."""
        widgets = self.rows.get(block_id)
        state = self.controller.block_state(block_id)
        if not widgets or state is None:
            return
        widgets["primary_slider"].set(state.primary_strength)
        widgets["primary_value"].configure(text=format_strength(state.primary_strength))
        if "donor_slider" in widgets:
            widgets["donor_slider"].set(state.donor_strength)
            widgets["donor_value"].configure(text=format_strength(state.donor_strength))

    def __update_summary(self):
        self.summary_label.configure(text=self.controller.bake_summary())

    def __status(self, message: str, error: bool = False):
        self.status_label.configure(text=message, text_color="#dc3545" if error else "#198754")

    # --- files ------------------------------------------------------------------------------------

    def __browse_primary(self):
        path = filedialog.askopenfilename(filetypes=[("LoRA", "*.safetensors")])
        if path:
            self.load_primary(path)

    def load_primary(self, path: str):
        error = self.controller.load_primary(path)
        _set(self.primary_entry, path)
        _set(self.donor_entry, self.controller.state.donor_path)
        _set(self.output_entry, self.controller.default_output_path())
        self.refresh()
        self.__status(error or "", error=bool(error))

    def __browse_donor(self):
        path = filedialog.askopenfilename(filetypes=[("LoRA", "*.safetensors")])
        if path:
            self.load_donor(path)

    def load_donor(self, path: str):
        error = self.controller.load_donor(path)
        _set(self.donor_entry, "" if error else path)
        self.refresh()
        self.__status(error or "", error=bool(error))

    def __clear_donor(self):
        self.controller.load_donor("")
        _set(self.donor_entry, "")
        self.refresh()

    def __save(self):
        error, message = self.controller.save(self.output_entry.get().strip())
        self.__status(error or message, error=bool(error))

    def __save_preset(self):
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                            filetypes=[("Preset", "*.json")])
        if path:
            error = self.controller.save_preset(path)
            self.__status(error or f"Preset saved to {os.path.basename(path)}", error=bool(error))

    def __load_preset(self):
        path = filedialog.askopenfilename(filetypes=[("Preset", "*.json")])
        if path:
            error = self.controller.load_preset(path)
            self.refresh()
            self.__status(error or f"Loaded {os.path.basename(path)}", error=bool(error))


def format_strength(value: float) -> str:
    return f"{value:+.2f}"


def _set(entry, text: str):
    entry.delete(0, "end")
    entry.insert(0, text or "")
