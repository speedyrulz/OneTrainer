"""What the sliders currently say, and the operations the quick-set buttons perform.

Kept apart from both the file and the UI so the same state can be saved as a preset, diffed, and
handed to the bake without any of those three knowing about the others.
"""

import json
from dataclasses import asdict, dataclass, field

# What a slider can be dragged to. Negative is meaningful — inverting a block's contribution is a
# real repair, not a mistake — and past about 3 the block stops being a contribution and starts being
# the whole image.
MIN_STRENGTH = -3.0
MAX_STRENGTH = 3.0


def clamp(value: float) -> float:
    return max(MIN_STRENGTH, min(MAX_STRENGTH, float(value)))


@dataclass
class BlockState:
    primary_enabled: bool = True
    primary_strength: float = 1.0
    donor_enabled: bool = True
    donor_strength: float = 0.0

    def contributes_primary(self) -> bool:
        return self.primary_enabled and abs(self.primary_strength) > 1e-9

    def contributes_donor(self) -> bool:
        return self.donor_enabled and abs(self.donor_strength) > 1e-9


@dataclass
class SliderState:
    """Every slider in the studio, plus which files they refer to."""

    blocks: dict[str, BlockState] = field(default_factory=dict)
    primary_path: str = ""
    donor_path: str = ""

    # --- construction ------------------------------------------------------------------------

    @classmethod
    def for_blocks(cls, block_ids, primary_path: str = "", donor_path: str = "") -> "SliderState":
        return cls(
            blocks={block_id: BlockState() for block_id in block_ids},
            primary_path=primary_path,
            donor_path=donor_path,
        )

    def copy(self) -> "SliderState":
        return SliderState(
            blocks={block_id: BlockState(**asdict(state)) for block_id, state in self.blocks.items()},
            primary_path=self.primary_path,
            donor_path=self.donor_path,
        )

    def resize_to(self, block_ids) -> "SliderState":
        """Keep the settings for blocks the new LoRA also has, and default the rest.

        Browsing a new LoRA auto-swaps rather than resetting, so switching between two checkpoints of
        the same training run keeps your edits.
        """
        wanted = list(block_ids)
        return SliderState(
            blocks={
                block_id: BlockState(**asdict(self.blocks[block_id]))
                if block_id in self.blocks else BlockState()
                for block_id in wanted
            },
            primary_path=self.primary_path,
            donor_path=self.donor_path,
        )

    # --- the quick-set buttons ---------------------------------------------------------------

    def set_strength(self, block_id: str, value: float, *, donor: bool = False) -> None:
        state = self.blocks.get(block_id)
        if state is None:
            return
        if donor:
            state.donor_strength = clamp(value)
        else:
            state.primary_strength = clamp(value)

    def zero(self, block_id: str, *, donor: bool = False) -> None:
        """[0] — take this block out. Disabled as well as zeroed, so the bake drops its keys."""
        state = self.blocks.get(block_id)
        if state is None:
            return
        if donor:
            state.donor_strength = 0.0
            state.donor_enabled = False
        else:
            state.primary_strength = 0.0
            state.primary_enabled = False

    def full(self, block_id: str, *, donor: bool = False) -> None:
        """[1] — back to how it was trained."""
        state = self.blocks.get(block_id)
        if state is None:
            return
        if donor:
            state.donor_strength = 1.0
            state.donor_enabled = True
        else:
            state.primary_strength = 1.0
            state.primary_enabled = True

    def invert(self, block_id: str, *, donor: bool = False) -> None:
        """[±] — flip the sign, so the block subtracts what it used to add."""
        state = self.blocks.get(block_id)
        if state is None:
            return
        if donor:
            state.donor_strength = clamp(-state.donor_strength)
        else:
            state.primary_strength = clamp(-state.primary_strength)

    def balance(self, block_id: str, *, from_donor: bool = False) -> None:
        """[⚖] — hold primary + donor at 1.0 for this block, which is how you cross-fade two LoRAs.

        Moving one side moves the other to whatever is left, so the block's total contribution stays
        put and only its *source* shifts.
        """
        state = self.blocks.get(block_id)
        if state is None:
            return
        if from_donor:
            state.primary_strength = clamp(1.0 - state.donor_strength)
            state.primary_enabled = True
        else:
            state.donor_strength = clamp(1.0 - state.primary_strength)
            state.donor_enabled = True

    # --- everything at once -------------------------------------------------------------------

    def apply_all(self, action: str, *, donor: bool = False, blocks=None) -> None:
        """Run one of the quick-sets over every block (or a named subset — e.g. one group)."""
        targets = list(blocks) if blocks is not None else list(self.blocks)
        for block_id in targets:
            match action:
                case "zero":
                    self.zero(block_id, donor=donor)
                case "full":
                    self.full(block_id, donor=donor)
                case "invert":
                    self.invert(block_id, donor=donor)
                case "balance":
                    self.balance(block_id, from_donor=donor)

    # --- reporting ----------------------------------------------------------------------------

    def is_default(self) -> bool:
        """Whether the bake would just copy the primary through unchanged."""
        return all(
            state.primary_enabled and state.primary_strength == 1.0 and not state.contributes_donor()
            for state in self.blocks.values()
        )

    def changed_blocks(self, other: "SliderState") -> list[str]:
        return [
            block_id for block_id, state in self.blocks.items()
            if other.blocks.get(block_id) != state
        ]

    # --- presets ------------------------------------------------------------------------------

    def to_json(self) -> dict:
        return {
            "blocks": {block_id: asdict(state) for block_id, state in self.blocks.items()},
            "primary_path": self.primary_path,
            "donor_path": self.donor_path,
        }

    @classmethod
    def from_json(cls, data: dict) -> "SliderState":
        blocks = {}
        for block_id, state in (data.get("blocks") or {}).items():
            known = {k: v for k, v in state.items() if k in BlockState.__dataclass_fields__}
            blocks[str(block_id)] = BlockState(**known)
        return cls(
            blocks=blocks,
            primary_path=str(data.get("primary_path", "")),
            donor_path=str(data.get("donor_path", "")),
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SliderState":
        with open(path, encoding="utf-8-sig") as f:
            return cls.from_json(json.load(f))
