"""Tests for the Repair Studio live preview: the patcher's math, and the engine's judgement.

The patcher is the piece that must be right: it applies a LoRA to a *loaded model* through forward
hooks, and its output has to equal what the bake would write to disk. Every claim is measured
against manually applied dense deltas on a tiny module tree — the Kronecker identity especially,
because index conventions read right and are wrong.

The engine's model loading needs real Krea 2 weights and is not run here; what is tested is its
judgement — which configs it accepts, what it refuses, and that the controller plumbs it sanely.

Run with:  pytest tests/test_repair_preview.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from modules.ui.RepairStudioTabController import RepairStudioTabController  # noqa: E402
from modules.util.repair import lora_file  # noqa: E402
from modules.util.repair.preview_patcher import PreviewPatcher, lokr_apply  # noqa: E402
from modules.util.repair.SliderState import SliderState  # noqa: E402

from safetensors.torch import save_file  # noqa: E402

# --- a tiny model with Flux-like block naming -----------------------------------------------------


class TinyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.img_attn_proj = torch.nn.Linear(6, 8)

    def forward(self, x):
        return self.img_attn_proj(x)


class TinyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.double_blocks = torch.nn.ModuleList([TinyBlock(), TinyBlock()])

    def forward(self, x):
        for block in self.double_blocks:
            x = torch.nn.functional.pad(block(x), (0, 6 - 8 % 6))[..., :6]
        return x


def make_transformer():
    torch.manual_seed(0)
    return TinyTransformer().eval()


def write_lora(path, *, rank=4, alpha=2.0, seed=1, blocks=(0, 1)):
    generator = torch.Generator().manual_seed(seed)
    state_dict = {}
    for i in blocks:
        name = f"lora_unet_double_blocks_{i}_img_attn_proj"
        state_dict[f"{name}.lora_down.weight"] = torch.randn(rank, 6, generator=generator)
        state_dict[f"{name}.lora_up.weight"] = torch.randn(8, rank, generator=generator)
        state_dict[f"{name}.alpha"] = torch.tensor(float(alpha))
    save_file(state_dict, str(path))
    return str(path)


def expected_output(transformer, lora, strengths: dict, x):
    """The ground truth: each block's dense delta applied by hand at its slider strength."""
    with torch.no_grad():
        out = x
        for i, block in enumerate(transformer.double_blocks):
            module = lora.modules[f"lora_unet_double_blocks_{i}_img_attn_proj"]
            strength = strengths.get(f"double_{i}", 1.0)
            y = block(out) + (out @ module.delta().T) * strength
            out = torch.nn.functional.pad(y, (0, 6 - 8 % 6))[..., :6]
        return out


# --- the Kronecker identity -----------------------------------------------------------------------


@pytest.mark.parametrize(("shape1", "shape2"), [
    ((3, 4), (2, 5)),
    ((1, 6), (7, 2)),
    ((4, 4), (4, 4)),
])
def test_lokr_apply_equals_the_dense_kronecker_product(shape1, shape2):
    torch.manual_seed(2)
    w1 = torch.randn(*shape1)
    w2 = torch.randn(*shape2)
    x = torch.randn(2, 3, shape1[1] * shape2[1])  # batch and sequence leading dims

    fast = lokr_apply(x, w1, w2)
    dense = x @ torch.kron(w1, w2).T

    assert torch.allclose(fast, dense, atol=1e-5)


# --- the hooks, measured --------------------------------------------------------------------------


def attach(transformer, lora_path, tmp_path=None):
    lora = lora_file.load(lora_path)
    state = SliderState.for_blocks(b.id for b in lora.blocks)
    patcher = PreviewPatcher()
    patcher.attach(transformer, lora, state)
    return lora, state, patcher


def test_hooked_output_equals_the_manual_delta(tmp_path):
    transformer = make_transformer()
    lora, state, patcher = attach(transformer, write_lora(tmp_path / "l.safetensors"))
    state.set_strength("double_0", 0.35)
    state.set_strength("double_1", -1.5)
    patcher.set_state(state)

    x = torch.randn(2, 6)
    with torch.no_grad():
        hooked = transformer(x)
    expected = expected_output(transformer_clean(tmp_path), lora,
                               {"double_0": 0.35, "double_1": -1.5}, x)

    assert torch.allclose(hooked, expected, atol=1e-5)


def transformer_clean(tmp_path):
    # same weights as make_transformer (same seed), no hooks
    return make_transformer()


def test_a_slider_change_takes_effect_with_no_reattach(tmp_path):
    transformer = make_transformer()
    lora, state, patcher = attach(transformer, write_lora(tmp_path / "l.safetensors"))
    x = torch.randn(2, 6)

    with torch.no_grad():
        at_full = transformer(x)
    state.set_strength("double_0", 0.0)
    state.blocks["double_0"].primary_enabled = True
    patcher.set_state(state)
    with torch.no_grad():
        at_zero = transformer(x)

    assert not torch.allclose(at_full, at_zero)
    assert torch.allclose(at_zero, expected_output(make_transformer(), lora,
                                                   {"double_0": 0.0}, x), atol=1e-5)


def test_a_disabled_block_contributes_nothing(tmp_path):
    transformer = make_transformer()
    lora, state, patcher = attach(transformer, write_lora(tmp_path / "l.safetensors"))
    state.zero("double_1")
    patcher.set_state(state)

    x = torch.randn(2, 6)
    with torch.no_grad():
        hooked = transformer(x)

    assert torch.allclose(hooked, expected_output(make_transformer(), lora,
                                                  {"double_1": 0.0}, x), atol=1e-5)


def test_detach_restores_the_model_exactly(tmp_path):
    transformer = make_transformer()
    x = torch.randn(2, 6)
    with torch.no_grad():
        before = transformer(x)

    _lora, _state, patcher = attach(transformer, write_lora(tmp_path / "l.safetensors"))
    patcher.detach()

    with torch.no_grad():
        after = transformer(x)
    # no weights were ever written, so the base model is bit-identical after detach
    assert torch.equal(before, after)


def test_a_lokr_file_hooks_without_densifying(tmp_path):
    generator = torch.Generator().manual_seed(3)
    path = tmp_path / "lokr.safetensors"
    name = "lora_unet_double_blocks_0_img_attn_proj"
    save_file({
        f"{name}.lokr_w1": torch.randn(4, 3, generator=generator),
        f"{name}.lokr_w2": torch.randn(2, 2, generator=generator),
        f"{name}.alpha": torch.tensor(2.0),
    }, str(path))

    transformer = make_transformer()
    lora, state, patcher = attach(transformer, str(path))
    state.set_strength("double_0", 0.5)
    patcher.set_state(state)

    x = torch.randn(2, 6)
    with torch.no_grad():
        hooked = transformer.double_blocks[0](x)
    module = lora.modules[name]
    expected = make_transformer().double_blocks[0](x) + (x @ module.delta().T) * 0.5

    assert torch.allclose(hooked, expected, atol=1e-5)
    assert all(p.dense is None for p in patcher._patches)  # never materialised


def test_kohya_and_comfy_namings_hook_the_same_layer(tmp_path):
    for naming, suffixes in (("lora_unet_double_blocks_0_img_attn_proj",
                              ("lora_down.weight", "lora_up.weight")),
                             ("diffusion_model.double_blocks.0.img_attn_proj",
                              ("lora_A.weight", "lora_B.weight"))):
        generator = torch.Generator().manual_seed(4)
        path = tmp_path / f"{suffixes[0][:6]}.safetensors"
        save_file({
            f"{naming}.{suffixes[0]}": torch.randn(4, 6, generator=generator),
            f"{naming}.{suffixes[1]}": torch.randn(8, 4, generator=generator),
        }, str(path))

        patcher = PreviewPatcher()
        lora = lora_file.load(str(path))
        patcher.attach(make_transformer(), lora,
                       SliderState.for_blocks(b.id for b in lora.blocks))

        assert patcher.matched == 1, naming
        assert patcher.unmatched == [], naming


def test_a_key_matching_nothing_is_reported_not_dropped(tmp_path):
    path = write_lora(tmp_path / "l.safetensors")
    # add a module the tiny model does not have
    from safetensors.torch import load_file
    state_dict = load_file(path)
    state_dict["lora_unet_single_blocks_9_linear1.lora_down.weight"] = torch.randn(4, 6)
    state_dict["lora_unet_single_blocks_9_linear1.lora_up.weight"] = torch.randn(8, 4)
    save_file(state_dict, path)

    patcher = PreviewPatcher()
    lora = lora_file.load(path)
    patcher.attach(make_transformer(), lora, SliderState.for_blocks(b.id for b in lora.blocks))

    assert patcher.matched == 2
    assert patcher.unmatched == ["lora_unet_single_blocks_9_linear1"]


def test_the_donor_side_reads_the_donor_sliders(tmp_path):
    transformer = make_transformer()
    lora = lora_file.load(write_lora(tmp_path / "d.safetensors"))
    state = SliderState.for_blocks(b.id for b in lora.blocks)
    state.blocks["double_0"].donor_strength = 0.25
    state.blocks["double_1"].donor_strength = 0.0

    patcher = PreviewPatcher("donor")
    patcher.attach(transformer, lora, state)

    x = torch.randn(2, 6)
    with torch.no_grad():
        hooked = transformer(x)

    assert torch.allclose(hooked, expected_output(make_transformer(), lora,
                                                  {"double_0": 0.25, "double_1": 0.0}, x),
                          atol=1e-5)


def test_the_hooked_delta_matches_what_the_bake_writes(tmp_path):
    """The preview's whole promise: what you see hooked is what the saved file does."""
    from modules.util.repair import bake as bake_module

    transformer = make_transformer()
    path = write_lora(tmp_path / "l.safetensors")
    lora, state, patcher = attach(transformer, path)
    state.set_strength("double_0", 0.6)
    state.zero("double_1")
    patcher.set_state(state)

    x = torch.randn(2, 6)
    with torch.no_grad():
        previewed = transformer(x)

    out_path = str(tmp_path / "baked.safetensors")
    bake_module.bake(lora, state, out_path, None)
    baked = lora_file.load(out_path)

    with torch.no_grad():
        clean = make_transformer()
        out = x
        for i, block in enumerate(clean.double_blocks):
            y = block(out)
            name = f"lora_unet_double_blocks_{i}_img_attn_proj"
            if name in baked.modules:
                y = y + out @ baked.modules[name].delta().T  # the file, at strength 1.0
            out = torch.nn.functional.pad(y, (0, 6 - 8 % 6))[..., :6]

    assert torch.allclose(previewed, out, atol=1e-5)


# --- the engine's judgement -----------------------------------------------------------------------


def make_config(model_type_name: str = "KREA_2", base_model: str = "some/model"):
    from modules.util.config.TrainConfig import TrainConfig
    from modules.util.enum.ModelType import ModelType

    config = TrainConfig.default_values()
    config.model_type = ModelType[model_type_name]
    config.base_model_name = base_model
    return config


def test_the_preview_admits_krea2_only():
    from modules.util.repair.preview_engine import Krea2PreviewEngine

    assert Krea2PreviewEngine.supports(make_config("KREA_2")) is None
    refusal = Krea2PreviewEngine.supports(make_config("STABLE_DIFFUSION_XL_10_BASE"))
    assert refusal is not None
    assert "Krea 2" in refusal


def test_the_preview_needs_a_base_model():
    from modules.util.repair.preview_engine import Krea2PreviewEngine

    refusal = Krea2PreviewEngine.supports(make_config("KREA_2", base_model=""))
    assert refusal is not None
    assert "base model" in refusal


def test_the_baseline_cache_is_keyed_on_the_settings():
    from modules.util.repair.preview_engine import PreviewSettings

    same = PreviewSettings(prompt="a cat", seed=1)
    also_same = PreviewSettings(prompt="a cat", seed=1)
    different = PreviewSettings(prompt="a cat", seed=2)

    assert same.key() == also_same.key()
    assert same.key() != different.key()


# --- the controller plumbing ----------------------------------------------------------------------


def test_without_a_train_config_the_tab_still_edits(tmp_path):
    controller = RepairStudioTabController()  # no config: preview off, everything else intact

    assert controller.load_primary(write_lora(tmp_path / "l.safetensors")) is None
    reason = controller.preview_supported()
    assert reason is not None
    assert "config" in reason


def test_the_wrong_model_type_reports_before_any_loading(tmp_path):
    controller = RepairStudioTabController(make_config("STABLE_DIFFUSION_15"))
    controller.load_primary(write_lora(tmp_path / "l.safetensors"))

    error = controller.load_preview()

    assert error is not None
    assert "Krea 2" in error
    assert controller.preview_engine is None  # nothing was constructed for a config it refuses


def test_loading_the_preview_without_a_lora_says_so():
    controller = RepairStudioTabController(make_config("KREA_2"))

    error = controller.load_preview()

    assert error is not None
    assert "Open a LoRA" in error


def test_render_without_a_loaded_model_raises_readably():
    from modules.util.repair.preview_engine import PreviewSettings

    controller = RepairStudioTabController(make_config("KREA_2"))

    with pytest.raises(RuntimeError, match="Load the preview model"):
        controller.render_preview(PreviewSettings())


def test_sync_preview_files_is_a_noop_until_loaded(tmp_path):
    controller = RepairStudioTabController(make_config("KREA_2"))
    controller.load_primary(write_lora(tmp_path / "l.safetensors"))

    assert controller.sync_preview_files() is None  # no engine, no crash


def test_close_preview_is_safe_to_repeat():
    controller = RepairStudioTabController(make_config("KREA_2"))

    controller.close_preview()
    controller.close_preview()
