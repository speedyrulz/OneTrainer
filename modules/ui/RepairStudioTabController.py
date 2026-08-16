"""The Repair Studio's logic, with no widgets in it.

Both UI frameworks drive this and read back from it, so what the tab does is testable without a
display and the two views cannot drift apart in behaviour — only in appearance.
"""

import os

from modules.util.repair import bake as bake_module
from modules.util.repair import lora_file
from modules.util.repair.block_discovery import Block
from modules.util.repair.lora_file import LoraFile, UnsupportedLoRA
from modules.util.repair.SliderState import MAX_STRENGTH, MIN_STRENGTH, SliderState

# The group headings, in the order a model runs them.
GROUP_LABELS = {
    "double": "Double blocks",
    "single": "Single blocks",
    "in": "Input blocks",
    "mid": "Middle block",
    "out": "Output blocks",
    "down": "Down blocks",
    "middle": "Mid block",
    "up": "Up blocks",
    "joint": "Joint blocks",
    "block": "Blocks",
    "te": "Text encoder",
    "other": "Other",
}


class RepairStudioTabController:
    def __init__(self, train_config=None):
        self.primary: LoraFile | None = None
        self.donor: LoraFile | None = None
        self.state = SliderState()
        self.error: str | None = None

        # the training config supplies the preview's base model; without one the tab still edits
        # and bakes, it just cannot render
        self.train_config = train_config
        self.preview_engine = None

    # --- loading ------------------------------------------------------------------------------

    def load_primary(self, path: str) -> str | None:
        """Open a LoRA to edit. Returns None, or a message explaining why it could not be opened."""
        if not path:
            self.primary = None
            self.state = SliderState()
            return None

        try:
            loaded = lora_file.load(path)
        except (UnsupportedLoRA, FileNotFoundError) as e:
            self.error = str(e)
            return self.error
        except Exception as e:
            self.error = f"Could not read {os.path.basename(path)}: {e}"
            return self.error

        self.error = None
        self.primary = loaded
        # Browsing a new LoRA auto-swaps rather than resetting: the settings for blocks the new file
        # also has are kept, so flipping between two checkpoints of one run does not cost the edits.
        self.state = self.state.resize_to(block.id for block in loaded.blocks)
        self.state.primary_path = path

        if self.donor is not None:
            mismatch = self.donor_mismatch()
            if mismatch:
                self.donor = None
                self.state.donor_path = ""
                return mismatch
        return None

    def load_donor(self, path: str) -> str | None:
        if not path:
            self.donor = None
            self.state.donor_path = ""
            return None

        try:
            loaded = lora_file.load(path)
        except (UnsupportedLoRA, FileNotFoundError) as e:
            return str(e)
        except Exception as e:
            return f"Could not read {os.path.basename(path)}: {e}"

        self.donor = loaded
        self.state.donor_path = path

        mismatch = self.donor_mismatch()
        if mismatch:
            self.donor = None
            self.state.donor_path = ""
            return mismatch
        return None

    def donor_mismatch(self) -> str | None:
        """Whether the donor can be blended with the primary at all.

        Two LoRAs from different base models share no module names, so a blend would silently
        produce a file that is just the two of them stapled together. Better to say so.
        """
        if self.primary is None or self.donor is None:
            return None

        shared = set(self.primary.modules) & set(self.donor.modules)
        if not shared:
            return (f"{self.donor.name} has no layers in common with {self.primary.name} — they were "
                    f"trained on different models, so there is nothing to blend.")

        for name in sorted(shared):
            primary_module = self.primary.modules[name]
            donor_module = self.donor.modules[name]
            if not (primary_module.is_standard and donor_module.is_standard):
                # a LoKR/LoHa module cannot be blended anyway (the bake refuses it exactly),
                # so shape-checking it here would only block loading a donor whose standard
                # blocks are perfectly usable
                continue
            if primary_module.up.shape[0] != donor_module.up.shape[0]:
                return (f"{self.donor.name} does not match {self.primary.name}: {name} is shaped "
                        f"{tuple(donor_module.up.shape)} against {tuple(primary_module.up.shape)}.")
        return None

    @property
    def has_donor(self) -> bool:
        return self.donor is not None

    # --- what the tab draws --------------------------------------------------------------------

    def blocks(self) -> list[Block]:
        return self.primary.blocks if self.primary is not None else []

    def grouped_blocks(self) -> list[tuple[str, list[Block]]]:
        """Blocks under their headings, in model order — the layout of the slider list."""
        groups: list[tuple[str, list[Block]]] = []
        for block in self.blocks():
            heading = GROUP_LABELS.get(block.group, block.group.title())
            if groups and groups[-1][0] == heading:
                groups[-1][1].append(block)
            else:
                groups.append((heading, [block]))
        return groups

    def block_state(self, block_id: str):
        return self.state.blocks.get(block_id)

    def module_count(self, block_id: str) -> int:
        if self.primary is None:
            return 0
        return self.primary.block_summary().get(block_id, 0)

    def slider_range(self) -> tuple[float, float]:
        return MIN_STRENGTH, MAX_STRENGTH

    # --- what the tab says ---------------------------------------------------------------------

    def primary_summary(self) -> str:
        if self.error:
            return self.error
        if self.primary is None:
            return "Open a LoRA to edit its blocks."

        blocks = len(self.blocks())
        summary = f"{self.primary.name} — {blocks} block(s), {len(self.primary.modules)} layers"

        low, high = self.primary.rank_range
        if high:
            summary += f", rank {low}" if low == high else f", rank {low}-{high}"

        kinds = sorted({m.lycoris_kind for m in self.primary.modules.values() if m.lycoris_kind})
        if kinds:
            names = {"lokr": "LoKR", "loha": "LoHa"}
            summary += ", " + "/".join(names.get(k, k) for k in kinds)
        return summary

    def donor_summary(self) -> str:
        if self.donor is None:
            return "No donor. Add one to blend blocks from a second LoRA."
        shared = len(set(self.primary.modules) & set(self.donor.modules)) if self.primary else 0
        return f"{self.donor.name} — {shared} layer(s) in common"

    def bake_summary(self) -> str:
        if self.primary is None:
            return ""
        return bake_module.preview(self.primary, self.state, self.donor).describe()

    def default_output_path(self) -> str:
        if self.primary is None:
            return ""
        base, ext = os.path.splitext(self.primary.path)
        return f"{base}-repaired{ext or '.safetensors'}"

    # --- doing things ---------------------------------------------------------------------------

    def apply(self, action: str, block_id: str | None = None, *, donor: bool = False) -> None:
        """One quick-set button, on one block or on all of them."""
        if block_id is None:
            self.state.apply_all(action, donor=donor)
            return
        match action:
            case "zero":
                self.state.zero(block_id, donor=donor)
            case "full":
                self.state.full(block_id, donor=donor)
            case "invert":
                self.state.invert(block_id, donor=donor)
            case "balance":
                self.state.balance(block_id, from_donor=donor)

    def set_strength(self, block_id: str, value: float, *, donor: bool = False) -> None:
        state = self.state.blocks.get(block_id)
        if state is None:
            return
        self.state.set_strength(block_id, value, donor=donor)
        # dragging a slider off zero means you want the block back
        if donor:
            state.donor_enabled = True
        else:
            state.primary_enabled = True

    def save(self, out_path: str) -> tuple[str | None, str]:
        """Bake to `out_path`. Returns (error or None, message to show)."""
        if self.primary is None:
            return "No LoRA is open.", ""
        if not out_path:
            return "Choose where to save the repaired LoRA.", ""

        if os.path.abspath(out_path) == os.path.abspath(self.primary.path):
            return (("That would overwrite the LoRA you are editing. Pick a different name so the "
                     "original survives a bake you did not mean."), "")

        try:
            summary = bake_module.bake(self.primary, self.state, out_path, self.donor)
        except UnsupportedLoRA as e:
            return str(e), ""
        except Exception as e:
            return f"Could not write {os.path.basename(out_path)}: {e}", ""

        return None, f"Saved {os.path.basename(out_path)} — {summary.describe()}"

    # --- presets ---------------------------------------------------------------------------------

    def save_preset(self, path: str) -> str | None:
        try:
            self.state.save(path)
        except Exception as e:
            return f"Could not save the preset: {e}"
        return None

    # --- the live preview ------------------------------------------------------------------------

    def preview_supported(self) -> str | None:
        """None when the preview can run, else the reason it cannot."""
        if self.train_config is None:
            return "The preview needs the training config for a base model."
        from modules.util.repair.preview_engine import Krea2PreviewEngine

        return Krea2PreviewEngine.supports(self.train_config)

    @property
    def preview_loaded(self) -> bool:
        return self.preview_engine is not None and self.preview_engine.loaded

    def load_preview(self) -> str | None:
        """Load the base model and hook the open files onto it. Slow — call off the UI thread."""
        reason = self.preview_supported()
        if reason:
            return reason
        if self.primary is None:
            return "Open a LoRA first — the preview shows the file being edited."

        if self.preview_engine is None:
            from modules.util.repair.preview_engine import Krea2PreviewEngine

            self.preview_engine = Krea2PreviewEngine(self.train_config)

        error = self.preview_engine.load()
        if error:
            return error
        return self.preview_engine.attach(self.primary, self.donor, self.state)

    def unload_preview(self) -> None:
        if self.preview_engine is not None:
            self.preview_engine.unload()

    def sync_preview_files(self) -> str | None:
        """Re-hook after the primary or donor changed. No-op while the model is not loaded."""
        if not self.preview_loaded:
            return None
        if self.primary is None:
            self.preview_engine.unload()
            return None
        return self.preview_engine.attach(self.primary, self.donor, self.state)

    def render_preview(self, settings):
        """(baseline, edited) as PIL images at the sliders' current strengths. Slow — off the UI
        thread. Raises with a readable message when it cannot render."""
        if not self.preview_loaded:
            raise RuntimeError("Load the preview model first.")
        baseline = self.preview_engine.render_baseline(self.state, settings)
        edited = self.preview_engine.render(self.state, settings)
        return baseline, edited

    def preview_match_summary(self) -> str:
        if self.preview_engine is None:
            return ""
        return self.preview_engine.match_summary()

    def close_preview(self) -> None:
        """Release everything the preview holds. Safe to call repeatedly."""
        if self.preview_engine is not None:
            self.preview_engine.unload()
            self.preview_engine = None

    def load_preset(self, path: str) -> str | None:
        try:
            loaded = SliderState.load(path)
        except Exception as e:
            return f"Could not read the preset: {e}"

        if self.primary is not None:
            # keep only what this LoRA has blocks for, so a preset from another model does not
            # populate the tab with sliders that control nothing
            loaded = loaded.resize_to(block.id for block in self.primary.blocks)
            loaded.primary_path = self.state.primary_path
            loaded.donor_path = self.state.donor_path
        self.state = loaded
        return None
