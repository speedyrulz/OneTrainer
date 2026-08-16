"""The Repair Studio tab, PySide6.

Structurally parallel to CtkRepairStudioTabView and sharing RepairStudioTabController with it, so
the two tabs behave identically and differ only in how they are drawn. See that file for why this
tab does not use the shared `components` abstraction.
"""

import os

from modules.ui.RepairStudioTabController import RepairStudioTabController

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# QSlider is integer-only, so strengths are held as hundredths and converted at the boundary.
SLIDER_SCALE = 100

QUICK_SETS = [
    ("0", "zero", "Take this block out entirely — its keys are dropped from the saved file."),
    ("1", "full", "Back to the strength it was trained at."),
    ("±", "invert", "Flip the sign, so the block subtracts what it used to add."),
    ("⚖", "balance", "Hold primary + donor at 1.0 for this block, to cross-fade the two LoRAs."),
]


class PySide6RepairStudioTabView(QWidget):
    # worker threads emit these; the connected slots run on the UI thread
    _preview_loaded = Signal(str)          # error message, or ""
    _preview_rendered = Signal(object, object, str)  # baseline, edited, error

    def __init__(self, parent, controller: RepairStudioTabController):
        super().__init__(parent)

        self.controller = controller
        self.rows: dict[str, dict] = {}
        self._render_busy = False
        self._render_pending = False

        self._preview_loaded.connect(self.__on_preview_loaded)
        self._preview_rendered.connect(self.__on_preview_rendered)

        self._auto_timer = QTimer(self)
        self._auto_timer.setSingleShot(True)
        self._auto_timer.setInterval(500)
        self._auto_timer.timeout.connect(self.__start_render)

        layout = QVBoxLayout(self)
        layout.addWidget(self.__build_header())

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        layout.addWidget(self.scroll, 1)

        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.addStretch(1)
        self.scroll.setWidget(self.container)

        layout.addWidget(self.__build_preview())
        layout.addWidget(self.__build_footer())
        self.refresh()

    # --- chrome --------------------------------------------------------------------------------

    def __build_header(self) -> QWidget:
        header = QFrame()
        header.setFrameShape(QFrame.Shape.StyledPanel)
        grid = QGridLayout(header)
        grid.setColumnStretch(1, 1)

        grid.addWidget(QLabel("LoRA"), 0, 0)
        self.primary_entry = QLineEdit()
        grid.addWidget(self.primary_entry, 0, 1)
        browse = QPushButton("Browse")
        browse.clicked.connect(self.__browse_primary)
        grid.addWidget(browse, 0, 2)

        self.primary_label = QLabel("")
        self.primary_label.setWordWrap(True)
        grid.addWidget(self.primary_label, 1, 1, 1, 2)

        grid.addWidget(QLabel("Donor"), 2, 0)
        self.donor_entry = QLineEdit()
        grid.addWidget(self.donor_entry, 2, 1)

        donor_buttons = QWidget()
        donor_row = QHBoxLayout(donor_buttons)
        donor_row.setContentsMargins(0, 0, 0, 0)
        donor_browse = QPushButton("Browse")
        donor_browse.clicked.connect(self.__browse_donor)
        donor_row.addWidget(donor_browse)
        donor_clear = QPushButton("Clear")
        donor_clear.clicked.connect(self.__clear_donor)
        donor_row.addWidget(donor_clear)
        grid.addWidget(donor_buttons, 2, 2)

        self.donor_label = QLabel("")
        self.donor_label.setWordWrap(True)
        grid.addWidget(self.donor_label, 3, 1, 1, 2)

        tools = QWidget()
        tools_row = QHBoxLayout(tools)
        tools_row.setContentsMargins(0, 0, 0, 0)
        tools_row.addWidget(QLabel("All blocks:"))
        for text, action, tooltip in QUICK_SETS:
            button = QPushButton(text)
            button.setFixedWidth(38)
            button.setToolTip(f"Every block: {tooltip}")
            button.clicked.connect(lambda _c=False, a=action: self.__quick_set_all(a))
            tools_row.addWidget(button)
        tools_row.addSpacing(16)

        load_preset = QPushButton("Load preset")
        load_preset.clicked.connect(self.__load_preset)
        tools_row.addWidget(load_preset)
        save_preset = QPushButton("Save preset")
        save_preset.clicked.connect(self.__save_preset)
        tools_row.addWidget(save_preset)
        tools_row.addStretch(1)
        grid.addWidget(tools, 4, 0, 1, 3)

        return header

    def __build_preview(self) -> QWidget:
        """The live render: baseline (as trained) beside the sliders' current meaning."""
        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        grid = QGridLayout(panel)
        grid.setColumnStretch(1, 1)

        buttons = QWidget()
        buttons_row = QHBoxLayout(buttons)
        buttons_row.setContentsMargins(0, 0, 0, 0)
        self.preview_load_button = QPushButton("Load preview model")
        self.preview_load_button.setToolTip(
            "Loads the model tab's base model so the sliders can render live. Krea 2 only for "
            "now, and it holds the model in VRAM until unloaded.")
        self.preview_load_button.clicked.connect(self.__load_preview_clicked)
        buttons_row.addWidget(self.preview_load_button)
        unload = QPushButton("Unload")
        unload.clicked.connect(self.__unload_preview_clicked)
        buttons_row.addWidget(unload)
        grid.addWidget(buttons, 0, 0)

        self.preview_status = QLabel("")
        self.preview_status.setWordWrap(True)
        grid.addWidget(self.preview_status, 0, 1, 1, 2)

        grid.addWidget(QLabel("Prompt"), 1, 0)
        self.preview_prompt = QLineEdit()
        grid.addWidget(self.preview_prompt, 1, 1, 1, 2)

        controls = QWidget()
        controls_row = QHBoxLayout(controls)
        controls_row.setContentsMargins(0, 0, 0, 0)
        self.preview_fields = {}
        for label, default, width in (("Seed", "42", 70), ("Steps", "20", 50),
                                      ("Width", "512", 60), ("Height", "512", 60)):
            controls_row.addWidget(QLabel(label))
            entry = QLineEdit(default)
            entry.setFixedWidth(width)
            controls_row.addWidget(entry)
            self.preview_fields[label.lower()] = entry
        render = QPushButton("Render")
        render.clicked.connect(self.__start_render)
        controls_row.addWidget(render)
        self.preview_auto = QCheckBox("Re-render on change")
        self.preview_auto.setToolTip("Render again half a second after a slider stops moving. Each "
                                     "render is a full sampling pass, so expect seconds, not frames.")
        controls_row.addWidget(self.preview_auto)
        controls_row.addStretch(1)
        grid.addWidget(controls, 2, 0, 1, 3)

        images = QWidget()
        images_row = QHBoxLayout(images)
        images_row.setContentsMargins(0, 0, 0, 0)
        self.preview_baseline_label = QLabel("baseline")
        self.preview_baseline_label.setFixedSize(300, 300)
        self.preview_baseline_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        images_row.addWidget(self.preview_baseline_label)
        self.preview_edited_label = QLabel("current sliders")
        self.preview_edited_label.setFixedSize(300, 300)
        self.preview_edited_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        images_row.addWidget(self.preview_edited_label)
        images_row.addStretch(1)
        grid.addWidget(images, 3, 0, 1, 3)

        return panel

    # --- preview plumbing ------------------------------------------------------------------------

    def __preview_status_set(self, message: str, error: bool = False):
        self.preview_status.setText(message)
        self.preview_status.setStyleSheet("color: #dc3545;" if error else "color: gray;")

    def __load_preview_clicked(self):
        reason = self.controller.preview_supported()
        if reason:
            self.__preview_status_set(reason, error=True)
            return
        self.__preview_status_set("Loading the preview model — this takes a while...")
        self.preview_load_button.setEnabled(False)

        def worker():
            error = self.controller.load_preview()
            self._preview_loaded.emit(error or "")

        import threading
        threading.Thread(target=worker, daemon=True).start()

    def __on_preview_loaded(self, error: str):
        self.preview_load_button.setEnabled(True)
        if error:
            self.__preview_status_set(error, error=True)
        else:
            summary = self.controller.preview_match_summary()
            self.__preview_status_set(f"Model loaded · {summary}" if summary else "Model loaded")

    def __unload_preview_clicked(self):
        self.controller.unload_preview()
        self.__preview_status_set("Preview model unloaded.")

    def __preview_settings(self):
        from modules.util.repair.preview_engine import PreviewSettings

        def integer(name, fallback):
            try:
                return int(self.preview_fields[name].text().strip())
            except (KeyError, ValueError):
                return fallback

        return PreviewSettings(
            prompt=self.preview_prompt.text(),
            seed=integer("seed", 42),
            steps=integer("steps", 20),
            width=integer("width", 512),
            height=integer("height", 512),
        )

    def __start_render(self):
        if not self.controller.preview_loaded:
            self.__preview_status_set("Load the preview model first.", error=True)
            return
        if self._render_busy:
            # remember that the sliders moved again; one more render follows the current one
            self._render_pending = True
            return
        self._render_busy = True
        self.__preview_status_set("Rendering...")
        settings = self.__preview_settings()

        def worker():
            try:
                baseline, edited = self.controller.render_preview(settings)
                self._preview_rendered.emit(baseline, edited, "")
            except Exception as e:
                self._preview_rendered.emit(None, None, str(e))

        import threading
        threading.Thread(target=worker, daemon=True).start()

    def __on_preview_rendered(self, baseline, edited, error: str):
        self._render_busy = False
        if error:
            self.__preview_status_set(error, error=True)
        else:
            for image, label in ((baseline, self.preview_baseline_label),
                                 (edited, self.preview_edited_label)):
                label.setPixmap(_pil_to_pixmap(image).scaled(
                    300, 300, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
            self.__preview_status_set("")
        if self._render_pending:
            self._render_pending = False
            self.__start_render()

    def __schedule_auto_render(self):
        if self.preview_auto.isChecked() and self.controller.preview_loaded:
            self._auto_timer.start()

    def __build_footer(self) -> QWidget:
        footer = QFrame()
        footer.setFrameShape(QFrame.Shape.StyledPanel)
        grid = QGridLayout(footer)
        grid.setColumnStretch(1, 1)

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        grid.addWidget(self.summary_label, 0, 0, 1, 3)

        grid.addWidget(QLabel("Save as"), 1, 0)
        self.output_entry = QLineEdit()
        grid.addWidget(self.output_entry, 1, 1)
        save = QPushButton("Save baked LoRA")
        save.setToolTip("Writes a normal .safetensors with the slider settings baked in. It loads "
                        "at strength 1.0 in ComfyUI and anywhere else — nothing has to know it was "
                        "edited.")
        save.clicked.connect(self.__save)
        grid.addWidget(save, 1, 2)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        grid.addWidget(self.status_label, 2, 0, 1, 3)

        return footer

    # --- the sliders ----------------------------------------------------------------------------

    def refresh(self):
        """Rebuild the slider list. Called when the LoRA changes, not when a slider moves."""
        self.rows.clear()
        while self.container_layout.count() > 1:
            item = self.container_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # deleteLater alone leaves the widget findable until the event loop runs, which
                # makes a refresh look like it did nothing
                widget.setParent(None)
                widget.deleteLater()

        self.primary_label.setText(self.controller.primary_summary())
        self.donor_label.setText(self.controller.donor_summary())

        index = 0
        for heading, blocks in self.controller.grouped_blocks():
            title = QLabel(heading)
            title.setStyleSheet("font-weight: bold;")
            self.container_layout.insertWidget(index, title)
            index += 1
            for block in blocks:
                self.container_layout.insertWidget(index, self.__build_row(block))
                index += 1

        if not self.controller.blocks():
            self.container_layout.insertWidget(0, QLabel("No LoRA open."))

        self.__update_summary()

    def __build_row(self, block) -> QWidget:
        low, high = self.controller.slider_range()
        state = self.controller.block_state(block.id)

        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        grid = QGridLayout(frame)
        grid.setColumnStretch(1, 1)

        count = self.controller.module_count(block.id)
        grid.addWidget(QLabel(f"{block.label}  ({count})"), 0, 0)

        widgets = {}
        widgets.update(self.__build_slider(grid, 0, block.id, low, high,
                                           state.primary_strength, donor=False))

        if self.controller.has_donor:
            donor_title = QLabel("donor")
            donor_title.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(donor_title, 1, 0)
            widgets.update(self.__build_slider(grid, 1, block.id, low, high,
                                               state.donor_strength, donor=True))

        self.rows[block.id] = widgets
        return frame

    def __build_slider(self, grid, row: int, block_id: str, low: float, high: float,
                       value: float, *, donor: bool) -> dict:
        prefix = "donor" if donor else "primary"

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(int(low * SLIDER_SCALE))
        slider.setMaximum(int(high * SLIDER_SCALE))
        slider.setValue(int(round(value * SLIDER_SCALE)))
        slider.valueChanged.connect(
            lambda v, b=block_id, d=donor: self.__on_slide(b, v / SLIDER_SCALE, donor=d))
        grid.addWidget(slider, row, 1)

        label = QLabel(format_strength(value))
        label.setFixedWidth(56)
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(label, row, 2)

        buttons = QWidget()
        button_row = QHBoxLayout(buttons)
        button_row.setContentsMargins(0, 0, 0, 0)
        for text, action, tooltip in QUICK_SETS:
            button = QPushButton(text)
            button.setFixedWidth(36)
            button.setToolTip(tooltip)
            button.clicked.connect(
                lambda _c=False, a=action, b=block_id, d=donor: self.__quick_set(b, a, donor=d))
            button_row.addWidget(button)
        grid.addWidget(buttons, row, 3)

        return {f"{prefix}_slider": slider, f"{prefix}_value": label}

    # --- reacting -------------------------------------------------------------------------------

    def __on_slide(self, block_id: str, value: float, *, donor: bool):
        self.controller.set_strength(block_id, value, donor=donor)
        self.__sync_row(block_id)
        self.__update_summary()
        self.__schedule_auto_render()

    def __quick_set(self, block_id: str, action: str, *, donor: bool):
        self.controller.apply(action, block_id, donor=donor)
        self.__sync_row(block_id)
        self.__update_summary()
        self.__schedule_auto_render()

    def __quick_set_all(self, action: str):
        self.controller.apply(action)
        for block_id in list(self.rows):
            self.__sync_row(block_id)
        self.__update_summary()
        self.__schedule_auto_render()

    def __sync_row(self, block_id: str):
        """Push the state back into the widgets, without rebuilding them."""
        widgets = self.rows.get(block_id)
        state = self.controller.block_state(block_id)
        if not widgets or state is None:
            return
        for prefix, value in (("primary", state.primary_strength),
                              ("donor", state.donor_strength)):
            slider = widgets.get(f"{prefix}_slider")
            if slider is None:
                continue
            # the setValue below re-fires valueChanged, which would write the rounded value straight
            # back over the state the quick-set just produced
            slider.blockSignals(True)
            slider.setValue(int(round(value * SLIDER_SCALE)))
            slider.blockSignals(False)
            widgets[f"{prefix}_value"].setText(format_strength(value))

    def __update_summary(self):
        self.summary_label.setText(self.controller.bake_summary())

    def __status(self, message: str, error: bool = False):
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {'#dc3545' if error else '#198754'};")

    # --- files ------------------------------------------------------------------------------------

    def __browse_primary(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open LoRA", "", "LoRA (*.safetensors)")
        if path:
            self.load_primary(path)

    def load_primary(self, path: str):
        error = self.controller.load_primary(path)
        self.primary_entry.setText(path)
        self.donor_entry.setText(self.controller.state.donor_path)
        self.output_entry.setText(self.controller.default_output_path())
        self.refresh()
        self.controller.sync_preview_files()
        self.__status(error or "", error=bool(error))

    def __browse_donor(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open donor LoRA", "", "LoRA (*.safetensors)")
        if path:
            self.load_donor(path)

    def load_donor(self, path: str):
        error = self.controller.load_donor(path)
        self.donor_entry.setText("" if error else path)
        self.refresh()
        self.controller.sync_preview_files()
        self.__status(error or "", error=bool(error))

    def __clear_donor(self):
        self.controller.load_donor("")
        self.donor_entry.setText("")
        self.refresh()
        self.controller.sync_preview_files()

    def __save(self):
        error, message = self.controller.save(self.output_entry.text().strip())
        self.__status(error or message, error=bool(error))

    def __save_preset(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save preset", "", "Preset (*.json)")
        if path:
            error = self.controller.save_preset(path)
            self.__status(error or f"Preset saved to {os.path.basename(path)}", error=bool(error))

    def __load_preset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load preset", "", "Preset (*.json)")
        if path:
            error = self.controller.load_preset(path)
            self.refresh()
            self.__status(error or f"Loaded {os.path.basename(path)}", error=bool(error))


def format_strength(value: float) -> str:
    return f"{value:+.2f}"


def _pil_to_pixmap(image) -> QPixmap:
    """PIL to QPixmap through PNG bytes — no dependency on PIL.ImageQt being importable."""
    import io as _io

    buffer = _io.BytesIO()
    image.save(buffer, format="PNG")
    pixmap = QPixmap()
    pixmap.loadFromData(buffer.getvalue(), "PNG")
    return pixmap
