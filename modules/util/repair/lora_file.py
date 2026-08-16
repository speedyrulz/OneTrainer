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


# Alpha values at or above this are a sentinel some tools (Fizgig, Comfy-Realtime-Lora) write into
# LyCORIS files to mean "the scale is already baked into the weights". Files like that exist in the
# wild; misreading one would rescale its whole delta by alpha/dim.
ALPHA_SENTINEL = 1e8


@dataclass
class LoraModule:
    """One adapted layer: its matrices, its scale, and anything else stored alongside.

    Covers the standard two-matrix form and the two LyCORIS forms whose delta is *linear in the
    first factor*, which is what makes exact per-block editing possible without SVD:

        LoKR:  delta = scale · kron(w1, w2)      (w1, w2 optionally factored as a @ b)
        LoHa:  delta = scale · (w1a@w1b ∘ w2a@w2b)
    """

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
    def lycoris_kind(self) -> str | None:
        """"lokr", "loha", or None — which LyCORIS form this module stores."""
        if any(name.startswith("lokr_") for name in self.tensors):
            return "lokr"
        if any(name.startswith("hada_") for name in self.tensors):
            return "loha"
        return None

    @property
    def is_tucker(self) -> bool:
        """A Tucker-decomposed module carries a core tensor the linear-factor bake cannot honour."""
        return any(name.startswith(("lokr_t", "hada_t")) for name in self.tensors)

    @property
    def is_dora(self) -> bool:
        return "dora_scale" in self.tensors

    def lycoris_complete(self) -> bool:
        """Whether both halves of the LyCORIS decomposition are actually in the file."""
        match self.lycoris_kind:
            case "lokr":
                w1 = self.tensors.get("lokr_w1") is not None or (
                    self.tensors.get("lokr_w1_a") is not None
                    and self.tensors.get("lokr_w1_b") is not None)
                w2 = self.tensors.get("lokr_w2") is not None or (
                    self.tensors.get("lokr_w2_a") is not None
                    and self.tensors.get("lokr_w2_b") is not None)
                return w1 and w2
            case "loha":
                return all(self.tensors.get(f"hada_w{i}_{half}") is not None
                           for i in (1, 2) for half in ("a", "b"))
            case _:
                return False

    def first_factor_name(self) -> str:
        """The tensor an exact per-block multiplier folds into.

        Both LyCORIS forms are linear in their first factor: kron(m·w1, w2) = m·kron(w1, w2) and
        (m·W1) ∘ W2 = m·(W1 ∘ W2). For standard LoRA it is `lora_up`, matching the existing bake.
        """
        if self.lycoris_kind == "lokr":
            return "lokr_w1_a" if "lokr_w1_a" in self.tensors else "lokr_w1"
        if self.lycoris_kind == "loha":
            return "hada_w1_a"
        return self.up_suffix()

    @property
    def rank(self) -> int:
        down = self.down
        return int(down.shape[0]) if down is not None else 0

    @property
    def alpha(self) -> float:
        """The stored alpha; without one, whatever value makes the scale come out as 1.0."""
        alpha = self.tensors.get("alpha")
        if alpha is None:
            return 1.0 if self.is_lycoris else float(self.rank)
        try:
            return float(alpha.item())
        except Exception:
            return 1.0 if self.is_lycoris else float(self.rank)

    @property
    def scale(self) -> float:
        """What inference multiplies this module's delta by.

        Standard LoRA: alpha / rank. LyCORIS: alpha over the smallest decomposition rank actually
        in use, mirroring LyCORIS's own inference modules — a divergent convention here silently
        rescales the whole file. A sentinel alpha means the scale is already in the weights.
        """
        kind = self.lycoris_kind
        if kind is None:
            return self.alpha / max(self.rank, 1)

        if self.alpha >= ALPHA_SENTINEL:
            return 1.0

        if kind == "loha":
            r1 = int(self.tensors["hada_w1_a"].shape[1])
            r2 = int(self.tensors["hada_w2_a"].shape[1])
            return self.alpha / max(1, min(r1, r2))

        # LoKR: the dims of whichever halves are factored; a full-matrix LoKR has dim 1
        dims = []
        if self.tensors.get("lokr_w1_a") is not None:
            dims.append(int(self.tensors["lokr_w1_a"].shape[1]))
        if self.tensors.get("lokr_w2_a") is not None:
            dims.append(int(self.tensors["lokr_w2_a"].shape[1]))
        return self.alpha / (max(1, min(dims)) if dims else 1)

    def down_suffix(self) -> str:
        return next(s for s in DOWN_SUFFIXES if s in self.tensors)

    def up_suffix(self) -> str:
        return next(s for s in UP_SUFFIXES if s in self.tensors)

    def _lokr_halves(self) -> tuple[torch.Tensor, torch.Tensor]:
        def half(full_name: str, a_name: str, b_name: str) -> torch.Tensor:
            full = self.tensors.get(full_name)
            if full is not None:
                return full.float()
            return self.tensors[a_name].float() @ self.tensors[b_name].float()

        return (half("lokr_w1", "lokr_w1_a", "lokr_w1_b"),
                half("lokr_w2", "lokr_w2_a", "lokr_w2_b"))

    def delta(self) -> torch.Tensor:
        """The weight change this module applies at strength 1.0, as a dense matrix.

        Only used to check a bake against what it replaced — it is the expensive form, which is the
        whole reason these decompositions exist. Convolution kernels are flattened into columns; a
        kron of the flattened halves equals the flattened kron, so the check stays exact.
        """
        match self.lycoris_kind:
            case "lokr":
                w1, w2 = self._lokr_halves()
                w1 = w1.reshape(w1.shape[0], -1)
                w2 = w2.reshape(w2.shape[0], -1)
                return torch.kron(w1, w2) * self.scale
            case "loha":
                first = (self.tensors["hada_w1_a"].float()
                         @ self.tensors["hada_w1_b"].float().reshape(self.tensors["hada_w1_b"].shape[0], -1))
                second = (self.tensors["hada_w2_a"].float()
                          @ self.tensors["hada_w2_b"].float().reshape(self.tensors["hada_w2_b"].shape[0], -1))
                return first * second * self.scale
            case _:
                up, down = self.up, self.down
                return (up.float().reshape(up.shape[0], -1)
                        @ down.float().reshape(down.shape[0], -1)) * self.scale


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

    tucker = [m.name for m in modules.values() if m.is_tucker]
    if tucker:
        raise UnsupportedLoRA(
            f"{os.path.basename(path)} uses Tucker-decomposed LoKR ({len(tucker)} module(s), "
            f"starting with {tucker[0]}). That form carries a core tensor the exact per-block bake "
            f"cannot honour, so the studio does not open it rather than hand you a file that is "
            f"quietly not what it says it is.")

    odd = [m.name for m in modules.values()
           if not m.is_standard and not (m.is_lycoris and m.lycoris_complete())]
    if odd:
        raise UnsupportedLoRA(
            f"{os.path.basename(path)} has {len(odd)} module(s) missing half of their "
            f"decomposition, starting with {odd[0]}. The file looks truncated.")

    return LoraFile(
        path=path,
        modules=modules,
        metadata=read_metadata(path),
        passthrough=passthrough,
    )
