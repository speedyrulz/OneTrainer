"""Applies an opened LoRA to a loaded model with live per-block strengths, for the preview.

The bake writes files; this makes the sliders visible. Each module the LoRA touches gets a forward
hook on the corresponding layer of the loaded model that adds the module's delta, scaled by whatever
the block's slider says *right now* — moving a slider changes the next render with no re-attaching
and no weight mutation. The base model's weights are never written, so detaching the hooks restores
it exactly.

Mapping file keys onto the loaded model reverses the export naming: kohya flattens the module path
under a `lora_unet_` prefix, ComfyUI prefixes the dotted path with `diffusion_model.`, and the peft
formats use the dotted path bare. All three are tried; a key that matches nothing is reported rather
than silently dropped, because a slider controlling only half its block is worse than an honest
count on screen.

LoKR is applied through the Kronecker identity — `(w1 ⊗ w2) @ x` computed from the factors — so no
module ever materialises its dense delta. LoHa has no such identity (the Hadamard product does not
factor through matmul), so LoHa files preview densely only if small, and are refused above a budget.
"""

import logging
from dataclasses import dataclass

from modules.util.repair.block_discovery import block_for
from modules.util.repair.lora_file import LoraFile, LoraModule
from modules.util.repair.SliderState import SliderState

import torch

logger = logging.getLogger(__name__)

# LoHa needs its dense delta materialised to be applied; past this total it is refused rather than
# quietly eating the VRAM the preview model needs.
LOHA_DENSE_BUDGET_GB = 1.0


def lokr_apply(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """`x @ kron(w1, w2).T` without building the Kronecker product.

    With w1 (o1, i1) and w2 (o2, i2), kron(w1, w2) maps an input laid out as (i1, i2) blocks to an
    output laid out as (o1, o2) blocks. Reshaping the last axis to (i1, i2) turns the big matmul
    into two small ones:

        y[..., a, b] = sum_{c,d} w1[a, c] · w2[b, d] · x[..., c, d]

    which is `w2` applied along the last axis and `w1` along the one before it. The tests pin this
    against the dense kron on small shapes, because index conventions are exactly the kind of thing
    that reads right and is wrong.
    """
    o1, i1 = w1.shape
    o2, i2 = w2.shape
    leading = x.shape[:-1]

    x = x.reshape(*leading, i1, i2)
    x = torch.einsum("...cd,bd->...cb", x, w2)
    x = torch.einsum("...cb,ac->...ab", x, w1)
    return x.reshape(*leading, o1 * o2)


@dataclass
class _Patch:
    """One hooked layer: how to compute its delta, and which block's slider scales it."""

    module: LoraModule
    block_id: str
    handle: object = None
    dense: torch.Tensor | None = None  # LoHa only

    def delta_for(self, x: torch.Tensor, target: torch.nn.Module) -> torch.Tensor:
        dtype = x.dtype
        device = x.device

        if self.dense is not None:
            return torch.nn.functional.linear(x, self.dense.to(device=device, dtype=dtype))

        if self.module.lycoris_kind == "lokr":
            w1, w2 = self.module._lokr_halves()
            return lokr_apply(x, w1.to(device=device, dtype=dtype),
                              w2.to(device=device, dtype=dtype)) * self.module.scale

        down = self.module.down.to(device=device, dtype=dtype)
        up = self.module.up.to(device=device, dtype=dtype)
        return torch.nn.functional.linear(
            torch.nn.functional.linear(x, down), up) * self.module.scale


class PreviewPatcher:
    """Hooks a LoraFile onto a model and scales every contribution by the live slider state.

    `side` picks which half of each block's state the hooks read: the primary LoRA's sliders or the
    donor's, so a donor file previews through a second patcher on the same model.
    """

    def __init__(self, side: str = "primary"):
        self.side = side
        self._patches: list[_Patch] = []
        self._strengths: dict[str, float] = {}
        self.matched = 0
        self.unmatched: list[str] = []

    # --- attaching -----------------------------------------------------------------------------

    def attach(self, transformer: torch.nn.Module, lora: LoraFile, state: SliderState) -> None:
        """Hook every module of `lora` that resolves to a layer of `transformer`."""
        self.detach()
        self.set_state(state)

        targets = _target_index(transformer)
        loha_bytes = 0

        for name, module in lora.modules.items():
            target = targets.get(_normalise_key(name))
            if target is None:
                self.unmatched.append(name)
                continue

            patch = _Patch(module=module, block_id=block_for(name).id)

            if module.lycoris_kind == "loha":
                dense = module.delta()  # scale included
                loha_bytes += dense.numel() * 2
                if loha_bytes > LOHA_DENSE_BUDGET_GB * 1024 ** 3:
                    raise RuntimeError(
                        f"previewing this LoHa needs more than {LOHA_DENSE_BUDGET_GB:g}GB of dense "
                        f"deltas (no factored way to apply a Hadamard product exists). The bake is "
                        f"unaffected — edit blind and check the saved file.")
                patch.dense = dense.to(torch.float16)

            patch.handle = target.register_forward_hook(self._hook_for(patch))
            self._patches.append(patch)

        self.matched = len(self._patches)
        if self.unmatched:
            logger.info("[preview] %d LoRA module(s) did not match a layer of the loaded model, "
                        "starting with %s", len(self.unmatched), self.unmatched[0])

    def _hook_for(self, patch: _Patch):
        def hook(target, args, output):
            strength = self._strengths.get(patch.block_id, 1.0)
            if strength == 0.0:
                return output
            x = args[0]
            return output + patch.delta_for(x, target) * strength

        return hook

    # --- live control ---------------------------------------------------------------------------

    def set_state(self, state: SliderState) -> None:
        """Adopt the sliders' current positions. Cheap: the next forward reads the new numbers."""
        if self.side == "donor":
            self._strengths = {
                block_id: (block.donor_strength if block.donor_enabled else 0.0)
                for block_id, block in state.blocks.items()
            }
        else:
            self._strengths = {
                block_id: (block.primary_strength if block.primary_enabled else 0.0)
                for block_id, block in state.blocks.items()
            }

    def detach(self) -> None:
        for patch in self._patches:
            if patch.handle is not None:
                patch.handle.remove()
        self._patches = []
        self.matched = 0
        self.unmatched = []

    @property
    def attached(self) -> bool:
        return bool(self._patches)


def _normalise_key(name: str) -> str:
    """A LoRA module key reduced to the dotted module path the model itself uses."""
    if name.startswith("diffusion_model."):
        name = name[len("diffusion_model."):]
    if name.startswith("lora_unet_"):
        # kohya flattening: dots became underscores; the exact-match index resolves the ambiguity
        return name[len("lora_unet_"):]
    return name


def _target_index(transformer: torch.nn.Module) -> dict[str, torch.nn.Module]:
    """Every hookable layer of the model, under every name an export might call it.

    Underscore flattening is ambiguous in general (`blocks_0_to_q` could split several ways), but
    matching against the model's *actual* module paths makes it exact: each real path is indexed
    both dotted and flattened, and a key either hits a real layer or it does not.
    """
    index: dict[str, torch.nn.Module] = {}
    for path, module in transformer.named_modules():
        # Linear only: the delta is applied with F.linear, which is wrong for a convolution, and
        # Krea 2 (like every DiT) has none. A conv key simply reports as unmatched.
        if not isinstance(module, torch.nn.Linear):
            continue
        index[path] = module
        index[path.replace(".", "_")] = module
    return index
