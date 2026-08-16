"""Reads a LoRA off disk into something the Repair Studio can edit.

A LoRA on disk is a flat pile of tensors whose names encode a structure. This puts the structure
back: keys grouped into modules, modules mapped to blocks, and the two matrices plus the scale that
make up each module pulled out under one set of names regardless of which trainer wrote them.

The four namings that turn up (OneTrainer writes all four; kohya-ss, ai-toolkit and ComfyUI write
subsets) differ only in the suffix and the separator:

    lora_unet_double_blocks_0_img_attn_proj.lora_down.weight   kohya
    diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight  ComfyUI / diffusers / peft

`lora_A` is `lora_down` and `lora_B` is `lora_up`; a file with no `.alpha` has its scale already
folded in, which is the same thing as alpha == rank.
"""

import os
from dataclasses import dataclass, field

from modules.util.repair.block_discovery import (
    LYCORIS_SUFFIXES,
    MODULE_SUFFIXES,
    Block,
    block_for,
    discover,
)

import torch

from safetensors import safe_open
from safetensors.torch import load_file

# The two names each of the LoRA matrices goes by, low-rank side first.
DOWN_SUFFIXES = ("lora_down.weight", "lora_A.weight")
UP_SUFFIXES = ("lora_up.weight", "lora_B.weight")


class UnsupportedLoRA(Exception):
    """Raised for a file the studio can read but must not pretend to have edited."""


def split_key(key: str) -> tuple[str, str] | None:
    """Split a tensor key into (module name, suffix), or None if it is not a LoRA key.

    Matched from the end rather than at the first dot: kohya's module names contain no dots, but
    ComfyUI's are nothing but dots.
    """
    for suffix in MODULE_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], suffix.lstrip(".")
    for suffix in LYCORIS_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], suffix.lstrip(".")
    return None


@dataclass
class LoraModule:
    """One adapted layer: its two matrices, its scale, and anything else stored alongside."""

    name: str
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)

    def _get(self, names) -> torch.Tensor | None:
        for candidate in names:
            tensor = self.tensors.get(candidate)
            if tensor is not None:
                return tensor
        return None

    @property
    def down(self) -> torch.Tensor | None:
        return self._get(DOWN_SUFFIXES)

    @property
    def up(self) -> torch.Tensor | None:
        return self._get(UP_SUFFIXES)

    @property
    def is_standard(self) -> bool:
        return self.down is not None and self.up is not None

    @property
    def is_lycoris(self) -> bool:
        return any(name.startswith(("lokr_", "hada_")) for name in self.tensors)

    @property
    def rank(self) -> int:
        down = self.down
        return int(down.shape[0]) if down is not None else 0

    @property
    def alpha(self) -> float:
        """The stored alpha, or the rank when there is none — both mean a scale of 1.0."""
        alpha = self.tensors.get("alpha")
        if alpha is None:
            return float(self.rank)
        try:
            return float(alpha.item())
        except Exception:
            return float(self.rank)

    @property
    def scale(self) -> float:
        """alpha / rank — what inference multiplies this module's output by."""
        return self.alpha / max(self.rank, 1)

    def down_suffix(self) -> str:
        return next(s for s in DOWN_SUFFIXES if s in self.tensors)

    def up_suffix(self) -> str:
        return next(s for s in UP_SUFFIXES if s in self.tensors)

    def delta(self) -> torch.Tensor:
        """The weight change this module applies at strength 1.0, as a dense matrix.

        Only used to check a bake against what it replaced — it is the expensive form, which is the
        whole reason LoRAs are stored as two matrices.
        """
        up, down = self.up, self.down
        return (up.float().reshape(up.shape[0], -1) @ down.float().reshape(down.shape[0], -1)) * self.scale


@dataclass
class LoraFile:
    """A LoRA read off disk, grouped and indexed by block."""

    path: str
    modules: dict[str, LoraModule]
    metadata: dict[str, str]
    passthrough: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def blocks(self) -> list[Block]:
        return discover(self.modules)

    def modules_in(self, block_id: str) -> list[str]:
        return sorted(name for name in self.modules if block_for(name).id == block_id)

    @property
    def rank_range(self) -> tuple[int, int]:
        ranks = [m.rank for m in self.modules.values() if m.is_standard]
        return (min(ranks), max(ranks)) if ranks else (0, 0)

    def block_summary(self) -> dict[str, int]:
        """How many modules sit under each block — what the tab shows beside each slider."""
        counts: dict[str, int] = {}
        for name in self.modules:
            block = block_for(name)
            counts[block.id] = counts.get(block.id, 0) + 1
        return counts


def read_metadata(path: str) -> dict[str, str]:
    try:
        with safe_open(path, framework="pt") as f:
            return {str(k): str(v) for k, v in (f.metadata() or {}).items()}
    except Exception:
        return {}


def load(path: str) -> LoraFile:
    """Read a .safetensors LoRA. Raises UnsupportedLoRA for anything that cannot be edited exactly."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    state_dict = load_file(path)

    modules: dict[str, LoraModule] = {}
    passthrough: dict[str, torch.Tensor] = {}

    for key, tensor in state_dict.items():
        split = split_key(key)
        if split is None:
            # bundled embeddings and anything else that is not part of the adapter: carried across
            # untouched so a repaired file keeps whatever the original shipped with
            passthrough[key] = tensor
            continue
        name, suffix = split
        modules.setdefault(name, LoraModule(name)).tensors[suffix] = tensor

    if not modules:
        raise UnsupportedLoRA(
            f"{os.path.basename(path)} has no LoRA weights in it — no lora_down/lora_up or "
            f"lora_A/lora_B keys were found.")

    lycoris = [m.name for m in modules.values() if m.is_lycoris]
    if lycoris:
        raise UnsupportedLoRA(
            f"{os.path.basename(path)} is a LyCORIS LoRA ({len(lycoris)} LoKR/LoHa modules). Those "
            f"cannot be rescaled per block without an SVD approximation, so the studio does not "
            f"open them rather than hand you a file that is quietly not what it says it is.")

    odd = [m.name for m in modules.values() if not m.is_standard]
    if odd:
        raise UnsupportedLoRA(
            f"{os.path.basename(path)} has {len(odd)} module(s) with only one of the two LoRA "
            f"matrices, starting with {odd[0]}. The file looks truncated.")

    return LoraFile(
        path=path,
        modules=modules,
        metadata=read_metadata(path),
        passthrough=passthrough,
    )
