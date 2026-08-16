"""Writes the edited LoRA out as a file that needs no special loader.

The arithmetic here is the point of the whole tab, so it is worth stating precisely. A LoRA module
contributes `scale · up @ down` to a weight, where `scale = alpha / rank`. Setting a block's slider
to *m* means contributing `m · scale · up @ down` instead.

That is baked by folding `m · scale` into `up` and then writing `alpha = rank`, which makes the new
file's own scale exactly 1.0. Nothing downstream has to know an edit happened: load it in ComfyUI at
strength 1.0 and you get what the slider said.

Blending in a donor uses **rank concatenation**, which is exact rather than an approximation:

    up   = cat([up_p · m_p · scale_p,  up_d · m_d · scale_d], dim=-1)
    down = cat([down_p,                down_d],               dim=0)

Multiplying those out gives `m_p·scale_p·up_p@down_p + m_d·scale_d·up_d@down_d` — the sum of the two
contributions, with no SVD and no loss. The price is a file whose rank is the sum of the two, which
is why a donor at zero strength is dropped rather than concatenated as a block of zeros.

LyCORIS (LoKR / LoHa) edits are exact too, in native format: both deltas are *linear in their first
factor* — `kron(m·w1, w2) = m·kron(w1, w2)` and `(m·W1) ∘ W2 = m·(W1 ∘ W2)` — so the multiplier
folds into `lokr_w1`/`lokr_w1_a`/`hada_w1_a` and nothing else changes. Alpha stays untouched (see
`_scaled_lycoris` for why that beats Fizgig's sentinel). The one thing that cannot be exact is
donor-blending a LyCORIS block, so that is refused rather than approximated.

Ported from [Fizgig](https://github.com/shootthesound/Fizgig) by Peter Neill (Apache 2.0).
"""

import json
import logging
import os
from dataclasses import dataclass, field

from modules.util.repair.block_discovery import block_for
from modules.util.repair.lora_file import LoraFile, LoraModule, UnsupportedLoRA
from modules.util.repair.SliderState import SliderState

import torch

from safetensors.torch import save_file

logger = logging.getLogger(__name__)

# Metadata that describes the file it came from rather than the file being written. Copying these
# across would leave every repaired LoRA claiming the source's content hash.
STALE_METADATA = ("sshs_model_hash", "sshs_legacy_hash", "modelspec.hash_sha256")

DROP, RESCALE, BLEND, KEEP, DONOR_ONLY = "drop", "rescale", "blend", "keep", "donor"


@dataclass
class BakeSummary:
    """What the bake did — or would do. Shown under the sliders and again after saving."""

    dropped: list[str] = field(default_factory=list)
    rescaled: list[str] = field(default_factory=list)
    blended: list[str] = field(default_factory=list)
    untouched: list[str] = field(default_factory=list)
    keys_in: int = 0
    keys_out: int = 0
    max_rank: int = 0
    out_path: str = ""

    def describe(self) -> str:
        parts = []
        if self.dropped:
            parts.append(f"{len(self.dropped)} block(s) removed")
        if self.rescaled:
            parts.append(f"{len(self.rescaled)} rescaled")
        if self.blended:
            parts.append(f"{len(self.blended)} blended with the donor")
        if not parts:
            return "unchanged — this would be a copy of the original"
        text = ", ".join(parts)
        if self.max_rank:
            text += f" · rank {self.max_rank}"
        if self.keys_out:
            text += f" · {self.keys_out} keys"
        return text


@dataclass
class _Decision:
    """What happens to one module, decided once and used by both the preview and the write."""

    name: str
    block_id: str
    action: str
    primary: LoraModule | None
    donor: LoraModule | None
    primary_strength: float = 1.0
    donor_strength: float = 0.0

    @property
    def rank(self) -> int:
        match self.action:
            case "blend":
                return self.primary.rank + self.donor.rank
            case "donor":
                return self.donor.rank
            case "drop":
                return 0
            case _:
                return self.primary.rank


def plan(primary: LoraFile, state: SliderState, donor: LoraFile | None = None) -> list[_Decision]:
    """Decide what happens to every module. The single source of truth for preview and bake alike."""
    donor_modules = donor.modules if donor is not None else {}
    decisions = []

    for name in sorted(set(primary.modules) | set(donor_modules)):
        block_id = block_for(name).id
        block = state.blocks.get(block_id)
        primary_module = primary.modules.get(name)
        donor_module = donor_modules.get(name)

        if block is None:
            # a block the sliders do not know about: leave the primary exactly as it was rather
            # than guessing at what the user would have wanted
            if primary_module is not None:
                decisions.append(_Decision(name, block_id, KEEP, primary_module, None))
            continue

        use_primary = block.contributes_primary() and primary_module is not None
        use_donor = block.contributes_donor() and donor_module is not None

        if not use_primary and not use_donor:
            action = DROP
        elif use_primary and use_donor:
            action = BLEND
        elif use_primary:
            # A LyCORIS or DoRA module at strength 1.0 passes through byte-identical, alpha and
            # all — LyCORIS because only the multiplier ever folds in, DoRA because folding its
            # scale would count as the edit its guard refuses. A plain standard module is
            # normalised to alpha = rank, so a non-unit scale is folded even at strength 1.0.
            if primary_module.is_lycoris or primary_module.is_dora:
                action = KEEP if block.primary_strength == 1.0 else RESCALE
            else:
                action = KEEP if (block.primary_strength == 1.0 and primary_module.scale == 1.0) \
                    else RESCALE
        else:
            action = DONOR_ONLY

        decisions.append(_Decision(
            name, block_id, action, primary_module, donor_module,
            block.primary_strength, block.donor_strength,
        ))

    return decisions


def summarise(decisions: list[_Decision], *, keys_out: int = 0, out_path: str = "") -> BakeSummary:
    by_action: dict[str, set[str]] = {}
    for decision in decisions:
        by_action.setdefault(decision.action, set()).add(decision.block_id)

    dropped = by_action.get(DROP, set())
    rescaled = (by_action.get(RESCALE, set()) | by_action.get(DONOR_ONLY, set())) - dropped
    blended = by_action.get(BLEND, set())
    untouched = by_action.get(KEEP, set()) - dropped - rescaled - blended

    ranks = [d.rank for d in decisions if d.action != DROP]

    return BakeSummary(
        dropped=sorted(dropped),
        rescaled=sorted(rescaled),
        blended=sorted(blended),
        untouched=sorted(untouched),
        keys_out=keys_out,
        max_rank=max(ranks) if ranks else 0,
        out_path=out_path,
    )


def preview(primary: LoraFile, state: SliderState, donor: LoraFile | None = None) -> BakeSummary:
    """What a bake would do, without writing anything."""
    return summarise(plan(primary, state, donor))


# --- doing it ------------------------------------------------------------------------------------


def _weighted(module: LoraModule, multiplier: float) -> tuple[torch.Tensor, torch.Tensor, int]:
    """One side's contribution, with its multiplier and its original scale folded into `up`."""
    up, down = module.up, module.down
    factor = multiplier * module.scale

    # the multiply happens in float32 so a small multiplier on an fp16 tensor does not round itself
    # away; factor == 1.0 skips it entirely so an untouched module comes out bit-identical
    new_up = up.clone() if factor == 1.0 else (up.to(torch.float32) * factor).to(up.dtype)

    return new_up, down.clone(), module.rank


def _guard_editable(module: LoraModule, block_id: str) -> None:
    """Refuse the edits whose result would not be what the slider says.

    A DoRA delta is renormalised against the base weight at load time, so scaling its matrices does
    not scale its effect linearly — the baked file would drift from the preview arithmetic. Dropping
    ([0]) and keeping ([1]) stay available; only in-between values are refused.
    """
    if module.is_dora:
        raise UnsupportedLoRA(
            f"{block_id} carries DoRA weights ({module.name}). A DoRA contribution does not scale "
            f"linearly, so a partial strength would not do what the slider says. Set this block to "
            f"0 or 1, or edit a non-DoRA export of the LoRA.")


def _scaled_lycoris(out: dict, module: LoraModule, multiplier: float) -> None:
    """Bake a multiplier into a LoKR/LoHa module in its native format — exact, no SVD.

    Both forms are linear in their first factor, so the multiplier folds into `lokr_w1` (or
    `lokr_w1_a` / `hada_w1_a`) the way the standard bake folds into `lora_up`:
    kron(m·w1, w2) = m·kron(w1, w2), and (m·W1) ∘ W2 = m·(W1 ∘ W2).

    Unlike the standard path, alpha is left exactly as it was. Folding the scale in too would need
    a sentinel alpha for loaders to interpret (Fizgig writes 1e10), and not every loader knows that
    convention — with alpha untouched, anything that read the original correctly reads the edit
    correctly, and an untouched module stays byte-identical.
    """
    first = module.first_factor_name()
    for suffix, tensor in module.tensors.items():
        if suffix == first:
            edited = (tensor.to(torch.float32) * multiplier).to(tensor.dtype)
            out[f"{module.name}.{suffix}"] = edited
        else:
            out[f"{module.name}.{suffix}"] = tensor


def _emit(out: dict, module: LoraModule, up: torch.Tensor, down: torch.Tensor, rank: int) -> None:
    """Write one module under the names it came in with, at an effective scale of exactly 1.0."""
    out[f"{module.name}.{module.up_suffix()}"] = up
    out[f"{module.name}.{module.down_suffix()}"] = down
    out[f"{module.name}.alpha"] = torch.tensor(float(rank))


def _blend(decision: _Decision) -> tuple[torch.Tensor, torch.Tensor, int]:
    up_p, down_p, rank_p = _weighted(decision.primary, decision.primary_strength)
    up_d, down_d, rank_d = _weighted(decision.donor, decision.donor_strength)

    if up_p.shape[0] != up_d.shape[0] or down_p.shape[1:] != down_d.shape[1:]:
        raise UnsupportedLoRA(
            f"The donor's {decision.name} is shaped {tuple(up_d.shape)} where the primary's is "
            f"{tuple(up_p.shape)}. The two LoRAs were trained on different models, so their blocks "
            f"cannot be blended.")

    # a donor saved in a different dtype would make cat fail
    common = up_p.dtype
    if up_d.dtype != common:
        up_d, down_d = up_d.to(common), down_d.to(common)
    if down_p.dtype != common:
        down_p = down_p.to(common)

    return (torch.cat([up_p, up_d], dim=-1),
            torch.cat([down_p, down_d], dim=0),
            rank_p + rank_d)


def bake(
        primary: LoraFile,
        state: SliderState,
        out_path: str,
        donor: LoraFile | None = None,
) -> BakeSummary:
    """Write the edited LoRA to `out_path` and report what changed."""
    decisions = plan(primary, state, donor)

    # bundled embeddings and anything else that was never part of the adapter travel with the file
    out: dict[str, torch.Tensor] = dict(primary.passthrough)

    for decision in decisions:
        match decision.action:
            case "drop":
                continue
            case "keep" if decision.primary_strength == 1.0:
                # untouched: the original keys, byte for byte, including a non-standard alpha
                for suffix, tensor in decision.primary.tensors.items():
                    out[f"{decision.name}.{suffix}"] = tensor
            case "blend":
                if decision.primary.is_lycoris or decision.donor.is_lycoris:
                    # rank concatenation needs the two-matrix form; blending a Kronecker or
                    # Hadamard block would mean a lossy SVD conversion, which this tab does not do
                    raise UnsupportedLoRA(
                        f"{decision.block_id} is a LoKR/LoHa block ({decision.name}), and blending "
                        f"one with a donor cannot be done exactly. Set one side of this block to "
                        f"zero, or use a standard-LoRA export for the blend.")
                _guard_editable(decision.primary, decision.block_id)
                _guard_editable(decision.donor, decision.block_id)
                _emit(out, decision.primary, *_blend(decision))
            case "donor":
                _guard_editable(decision.donor, decision.block_id)
                if decision.donor.is_lycoris:
                    _scaled_lycoris(out, decision.donor, decision.donor_strength)
                else:
                    _emit(out, decision.donor, *_weighted(decision.donor, decision.donor_strength))
            case _:
                _guard_editable(decision.primary, decision.block_id)
                if decision.primary.is_lycoris:
                    _scaled_lycoris(out, decision.primary, decision.primary_strength)
                else:
                    _emit(out, decision.primary,
                          *_weighted(decision.primary, decision.primary_strength))

    summary = summarise(decisions, keys_out=len(out), out_path=out_path)
    summary.keys_in = (sum(len(m.tensors) for m in primary.modules.values())
                       + len(primary.passthrough)
                       + (sum(len(m.tensors) for m in donor.modules.values()) if donor else 0))

    _write(out, primary, state, donor, summary)
    logger.info("repair studio: %s -> %s", summary.describe(), out_path)
    return summary


def _write(out: dict, primary: LoraFile, state: SliderState, donor: LoraFile | None,
           summary: BakeSummary) -> None:
    metadata = dict(primary.metadata)
    for key in STALE_METADATA:
        metadata.pop(key, None)

    if summary.max_rank:
        # external tools read these to report what a LoRA is, and a donor blend changes both
        metadata["ss_network_dim"] = str(summary.max_rank)
        metadata["ss_network_alpha"] = str(float(summary.max_rank))

    try:
        metadata["ot_repair_studio"] = json.dumps(state.to_json(), separators=(",", ":"))
    except (TypeError, ValueError):
        logger.warning("could not record the slider state in the file's metadata", exc_info=True)
    if donor is not None:
        metadata["ot_repair_studio_donor"] = os.path.basename(donor.path)

    directory = os.path.dirname(os.path.abspath(summary.out_path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    # safetensors will not write a non-contiguous view
    save_file({k: v.contiguous() for k, v in out.items()}, summary.out_path, metadata=metadata)


def effective_delta(module: LoraModule, multiplier: float) -> torch.Tensor:
    """What this module contributes to its weight at the given slider position.

    Exists so a test can check a baked file against the thing it is supposed to be equivalent to,
    which is the only claim in this file worth verifying by measurement rather than by reading.
    Works for every form the studio edits: `delta()` densifies standard, LoKR and LoHa alike.
    """
    return module.delta() * multiplier
