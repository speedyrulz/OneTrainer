"""Works out which transformer block each LoRA key belongs to, whatever named it.

Fizgig's Repair Studio has thirty-two sliders because Klein 9B has thirty-two blocks and Fizgig
trains one model family. OneTrainer trains fifteen-odd architectures whose block layouts have nothing
in common — an SDXL UNet is input/middle/output blocks, Flux is 19 double plus 38 single, Qwen is a
flat stack — so the blocks are *discovered from the file* instead of declared, and the tab builds as
many sliders as the LoRA turns out to have.

Only the key names are read. That means a LoRA trained anywhere OneTrainer can read (its own four
formats, kohya-ss, ai-toolkit, ComfyUI) resolves to the same block ids, and an architecture nobody has
thought of yet still lands somewhere sensible rather than failing.
"""

import re
from dataclasses import dataclass

# Every LoRA key ends in one of these. What comes before is the module.
MODULE_SUFFIXES = (
    ".lora_down.weight", ".lora_up.weight",     # kohya
    ".lora_A.weight", ".lora_B.weight",         # diffusers / peft, ComfyUI, OneTrainer's ORIGINAL
    ".alpha", ".dora_scale",
)

# LyCORIS. Recognised so a LoKR/LoHa file is refused with a straight answer rather than silently
# passed through as if it had been edited.
LYCORIS_SUFFIXES = (
    ".lokr_w1", ".lokr_w2", ".lokr_w1_a", ".lokr_w1_b", ".lokr_w2_a", ".lokr_w2_b",
    ".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b",
)

# The text-encoder namespaces every format uses, so a text-encoder key is never mistaken for a
# denoiser one (both end in "layers_N" often enough to matter).
_TEXT_ENCODER_RE = re.compile(r"(?:^|_|\.)(?:lora_)?te(\d*)_|text_encoders?[._]")

_GROUP_ORDER = {
    "double": 10, "single": 20,      # Flux-style two-stream then single-stream
    "in": 10, "mid": 20, "out": 30,  # sgm UNet, in model order
    "down": 10, "middle": 20, "up": 30,   # diffusers UNet, likewise
    "joint": 10, "block": 10,        # flat stacks
    "te": 90,                        # text encoders after the denoiser
    "other": 99,
}


@dataclass(frozen=True)
class Block:
    """One slider: an id, what to show on it, and where it sits in the model."""

    id: str
    label: str
    group: str
    index: int

    @property
    def sort_key(self) -> tuple:
        return (_GROUP_ORDER.get(self.group, 50), self.group, self.index, self.id)


# Ordered, and the order is the whole trick: an SDXL key is
# `input_blocks_4_1_transformer_blocks_0_attn1_to_q`, which matches *both* the sgm rule and the flat
# `transformer_blocks_N` rule. The outer structure is the one a user means by "block", so the
# coarse-grained patterns are tried first and the flat stack is the last resort.
_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Flux / Chroma / HunyuanVideo: two-stream then single-stream
    (re.compile(r"double_blocks_(\d+)"), "double"),
    (re.compile(r"single_blocks_(\d+)"), "single"),

    # sgm UNet (SD 1.5, SDXL as kohya writes it). The middle block captures nothing on purpose:
    # `middle_block_1` is the attention *inside* the one middle block, not middle block number one,
    # so capturing it would split a single block across three sliders.
    (re.compile(r"input_blocks_(\d+)"), "in"),
    (re.compile(r"middle_block"), "mid"),
    (re.compile(r"output_blocks_(\d+)"), "out"),

    # diffusers UNet, likewise
    (re.compile(r"down_blocks_(\d+)"), "down"),
    (re.compile(r"mid_block"), "middle"),
    (re.compile(r"up_blocks_(\d+)"), "up"),

    # SD3-style joint blocks
    (re.compile(r"joint_blocks_(\d+)"), "joint"),

    # flat DiT stacks: Qwen, Sana, PixArt, Z-Image, Krea 2, Flux 2 ...
    (re.compile(r"transformer_blocks_(\d+)"), "block"),
    (re.compile(r"(?:^|_)blocks_(\d+)"), "block"),
    (re.compile(r"(?:^|_)layers_(\d+)"), "block"),
]

_TE_LAYER_RE = re.compile(r"layers_(\d+)")


def normalise(module_name: str) -> str:
    """Flatten a module name so one set of patterns reads every format.

    ComfyUI and diffusers write `double_blocks.0.img_attn.proj`; kohya writes
    `lora_unet_double_blocks_0_img_attn_proj`. Turning dots into underscores makes them the same
    string, which is why there is one pattern table rather than one per format.
    """
    return module_name.replace(".", "_")


def block_for(module_name: str) -> Block:
    """Which block a module belongs to. Never returns None — anything unrecognised lands in `other`.

    A key with no block of its own still has to be reachable: dropping it silently would mean a
    slider set to zero that quietly kept contributing.
    """
    flat = normalise(module_name)

    te = _TEXT_ENCODER_RE.search(flat)
    if te:
        which = te.group(1) or "1"
        layer = _TE_LAYER_RE.search(flat)
        if layer:
            index = int(layer.group(1))
            return Block(f"te{which}_{index}", f"Text encoder {which} · layer {index}", "te", index)
        return Block(f"te{which}", f"Text encoder {which}", "te", -1)

    for pattern, group in _PATTERNS:
        match = pattern.search(flat)
        if match:
            raw = match.group(1) if match.lastindex else ""
            index = int(raw) if raw else 0
            return Block(f"{group}_{index}", _label_for(group, index), group, index)

    return Block("other", "Everything else", "other", 0)


_GROUP_LABELS = {
    "double": "Double block",
    "single": "Single block",
    "in": "Input block",
    "mid": "Middle block",
    "out": "Output block",
    "down": "Down block",
    "middle": "Mid block",
    "up": "Up block",
    "joint": "Joint block",
    "block": "Block",
}


def _label_for(group: str, index: int) -> str:
    name = _GROUP_LABELS.get(group, group.title())
    if group in ("mid", "middle"):
        return name
    return f"{name} {index}"


def discover(module_names) -> list[Block]:
    """Every distinct block these modules touch, in model order."""
    blocks = {}
    for name in module_names:
        block = block_for(name)
        blocks[block.id] = block
    return sorted(blocks.values(), key=lambda b: b.sort_key)
