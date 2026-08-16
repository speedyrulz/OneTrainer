"""Tests for the Repair Studio: block discovery, the slider state, and the bake.

The bake makes one claim, and it is the only reason to trust the tab: a saved file loaded at
strength 1.0 contributes exactly what the sliders said it would. That is checked by measurement —
multiplying the baked matrices out and comparing against the weighted sum they replaced — rather than
by reading the code, because "exact" is the kind of word that needs evidence.

Run with:  pytest tests/test_repair_studio.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from modules.ui.RepairStudioTabController import RepairStudioTabController  # noqa: E402
from modules.util.repair import bake as bake_module  # noqa: E402
from modules.util.repair import lora_file  # noqa: E402
from modules.util.repair.block_discovery import block_for, discover  # noqa: E402
from modules.util.repair.lora_file import UnsupportedLoRA  # noqa: E402
from modules.util.repair.SliderState import SliderState  # noqa: E402

from safetensors.torch import load_file, save_file  # noqa: E402

# --- fixtures ------------------------------------------------------------------------------------

FLUX_MODULES = [
    "lora_unet_double_blocks_0_img_attn_proj",
    "lora_unet_double_blocks_1_txt_mlp_0",
    "lora_unet_single_blocks_0_linear1",
    "lora_unet_single_blocks_11_linear2",
]


def write_lora(path, module_names, *, rank=4, alpha=2.0, out=8, inp=6, seed=0,
               suffixes=("lora_down.weight", "lora_up.weight"), metadata=None):
    generator = torch.Generator().manual_seed(seed)
    down_suffix, up_suffix = suffixes
    state_dict = {}
    for name in module_names:
        state_dict[f"{name}.{down_suffix}"] = torch.randn(rank, inp, generator=generator)
        state_dict[f"{name}.{up_suffix}"] = torch.randn(out, rank, generator=generator)
        if alpha is not None:
            state_dict[f"{name}.alpha"] = torch.tensor(float(alpha))
    save_file(state_dict, str(path), metadata=metadata)
    return str(path)


@pytest.fixture
def flux_lora(tmp_path):
    return write_lora(tmp_path / "primary.safetensors", FLUX_MODULES, seed=1)


@pytest.fixture
def donor_lora(tmp_path):
    return write_lora(tmp_path / "donor.safetensors", FLUX_MODULES, seed=2, rank=2, alpha=1.0)


def contribution(module, multiplier: float):
    return bake_module.effective_delta(module, multiplier)


# --- block discovery -----------------------------------------------------------------------------


@pytest.mark.parametrize(("module", "expected"), [
    # Flux / Chroma / HunyuanVideo, kohya naming
    ("lora_unet_double_blocks_7_img_attn_proj", "double_7"),
    ("lora_unet_single_blocks_23_linear1", "single_23"),
    # the same model out of ComfyUI, which uses dots
    ("diffusion_model.double_blocks.7.img_attn.proj", "double_7"),
    # sgm UNet (SD 1.5 / SDXL as kohya writes it)
    ("lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q", "in_4"),
    ("lora_unet_middle_block_1_transformer_blocks_0_attn2_to_k", "mid_0"),
    ("lora_unet_output_blocks_5_1_transformer_blocks_1_ff_net_0_proj", "out_5"),
    # diffusers UNet
    ("lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_v", "down_1"),
    ("lora_unet_up_blocks_2_attentions_1_proj_out", "up_2"),
    # SD3
    ("lora_unet_joint_blocks_12_x_block_attn_qkv", "joint_12"),
    # flat DiT stacks
    ("lora_unet_transformer_blocks_9_attn_to_q", "block_9"),
    ("lora_unet_blocks_3_ff_net_2", "block_3"),
])
def test_a_key_resolves_to_the_block_a_user_would_call_it(module, expected):
    assert block_for(module).id == expected


def test_the_outer_structure_wins_over_the_inner_one():
    # an SDXL key contains "transformer_blocks_0" inside "input_blocks_4"; the outer one is what
    # anyone means by "block", and picking the inner one would collapse the whole UNet onto ~10 ids
    assert block_for("lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q").id == "in_4"


@pytest.mark.parametrize("module", [
    "lora_te1_text_model_encoder_layers_5_self_attn_q_proj",
    "lora_te2_text_model_encoder_layers_5_mlp_fc1",
    "text_encoders.clip_l.transformer.text_model.encoder.layers.5.self_attn.k_proj",
])
def test_text_encoder_layers_are_not_mistaken_for_denoiser_blocks(module):
    block = block_for(module)
    assert block.group == "te"
    assert block.index == 5


def test_anything_unrecognised_is_still_reachable():
    # a key with no slider would be a block silently contributing after being "removed"
    block = block_for("lora_unet_final_layer_adaLN_modulation_1")
    assert block.id == "other"


def test_blocks_come_back_in_model_order():
    modules = ["lora_unet_single_blocks_2_linear1", "lora_unet_double_blocks_1_img_mod_lin",
               "lora_unet_double_blocks_0_img_mod_lin", "lora_te1_text_model_encoder_layers_0_mlp_fc1"]

    assert [b.id for b in discover(modules)] == ["double_0", "double_1", "single_2", "te1_0"]


def test_ten_sorts_after_nine_not_after_one():
    modules = [f"lora_unet_double_blocks_{i}_img_mod_lin" for i in (1, 9, 10, 2)]

    assert [b.index for b in discover(modules)] == [1, 2, 9, 10]


# --- reading the file ----------------------------------------------------------------------------


def test_a_kohya_lora_is_read(flux_lora):
    loaded = lora_file.load(flux_lora)

    assert len(loaded.modules) == len(FLUX_MODULES)
    assert [b.id for b in loaded.blocks] == ["double_0", "double_1", "single_0", "single_11"]


def test_a_peft_style_lora_is_read_the_same_way(tmp_path):
    path = write_lora(tmp_path / "peft.safetensors", FLUX_MODULES,
                      suffixes=("lora_A.weight", "lora_B.weight"))

    loaded = lora_file.load(path)

    assert len(loaded.modules) == len(FLUX_MODULES)
    assert loaded.modules[FLUX_MODULES[0]].is_standard


def test_a_dotted_comfyui_name_is_not_split_at_the_first_dot(tmp_path):
    name = "diffusion_model.double_blocks.0.img_attn.proj"
    path = write_lora(tmp_path / "comfy.safetensors", [name],
                      suffixes=("lora_A.weight", "lora_B.weight"))

    loaded = lora_file.load(path)

    assert list(loaded.modules) == [name]


def test_a_file_with_no_alpha_is_already_at_scale_one(tmp_path):
    path = write_lora(tmp_path / "no_alpha.safetensors", FLUX_MODULES, alpha=None)

    module = lora_file.load(path).modules[FLUX_MODULES[0]]

    assert module.scale == 1.0


def test_the_scale_is_alpha_over_rank(flux_lora):
    module = lora_file.load(flux_lora).modules[FLUX_MODULES[0]]

    assert module.rank == 4
    assert module.alpha == 2.0
    assert module.scale == 0.5


def test_a_lokr_lora_now_opens(tmp_path):
    path = tmp_path / "lokr.safetensors"
    save_file({
        "lora_unet_double_blocks_0_img_attn_proj.lokr_w1": torch.randn(4, 4),
        "lora_unet_double_blocks_0_img_attn_proj.lokr_w2": torch.randn(4, 4),
    }, str(path))

    loaded = lora_file.load(str(path))

    module = loaded.modules["lora_unet_double_blocks_0_img_attn_proj"]
    assert module.lycoris_kind == "lokr"
    assert not module.is_standard


def test_a_file_that_is_not_a_lora_says_so(tmp_path):
    path = tmp_path / "nope.safetensors"
    save_file({"model.diffusion_model.weight": torch.randn(4, 4)}, str(path))

    with pytest.raises(UnsupportedLoRA, match="no LoRA weights"):
        lora_file.load(str(path))


def test_a_half_written_module_is_caught(tmp_path):
    path = tmp_path / "truncated.safetensors"
    save_file({"lora_unet_double_blocks_0_img_attn_proj.lora_down.weight": torch.randn(4, 6)},
              str(path))

    with pytest.raises(UnsupportedLoRA, match="truncated"):
        lora_file.load(str(path))


# --- the bake, measured --------------------------------------------------------------------------


def bake_with(tmp_path, primary_path, edits: dict, donor_path=None, donor_edits=None):
    primary = lora_file.load(primary_path)
    donor = lora_file.load(donor_path) if donor_path else None
    state = SliderState.for_blocks(b.id for b in primary.blocks)
    for block_id, value in edits.items():
        state.blocks[block_id].primary_strength = value
    for block_id, value in (donor_edits or {}).items():
        state.blocks[block_id].donor_strength = value

    out = str(tmp_path / "baked.safetensors")
    summary = bake_module.bake(primary, state, out, donor)
    return primary, donor, lora_file.load(out), summary


def test_a_rescaled_block_contributes_exactly_what_the_slider_said(tmp_path, flux_lora):
    primary, _donor, baked, _summary = bake_with(tmp_path, flux_lora, {"double_0": 0.35})

    for name in primary.modules_in("double_0"):
        want = contribution(primary.modules[name], 0.35)
        got = contribution(baked.modules[name], 1.0)
        assert torch.allclose(want, got, atol=1e-5), name


def test_the_baked_file_runs_at_strength_one(tmp_path, flux_lora):
    _primary, _donor, baked, _summary = bake_with(tmp_path, flux_lora, {"double_0": 0.35})

    # alpha == rank, so ComfyUI and everything else multiply by exactly 1.0 and the sliders are
    # the file rather than something you have to remember to reapply
    for module in baked.modules.values():
        assert module.scale == 1.0


def test_an_inverted_block_subtracts_what_it_used_to_add(tmp_path, flux_lora):
    primary, _donor, baked, _summary = bake_with(tmp_path, flux_lora, {"single_0": -1.0})

    for name in primary.modules_in("single_0"):
        want = -contribution(primary.modules[name], 1.0)
        assert torch.allclose(want, contribution(baked.modules[name], 1.0), atol=1e-5)


def test_a_zeroed_block_is_gone_from_the_file(tmp_path, flux_lora):
    primary = lora_file.load(flux_lora)
    state = SliderState.for_blocks(b.id for b in primary.blocks)
    state.zero("double_0")

    out = str(tmp_path / "baked.safetensors")
    summary = bake_module.bake(primary, state, out, None)
    baked = lora_file.load(out)

    # not "written as zeros" — actually absent, so the file gets smaller
    assert not any(block_for(name).id == "double_0" for name in baked.modules)
    assert summary.dropped == ["double_0"]


def test_an_untouched_block_comes_out_byte_identical(tmp_path, flux_lora):
    path = write_lora(tmp_path / "unit_alpha.safetensors", FLUX_MODULES, rank=4, alpha=4.0)
    primary = lora_file.load(path)
    state = SliderState.for_blocks(b.id for b in primary.blocks)

    out = str(tmp_path / "baked.safetensors")
    bake_module.bake(primary, state, out, None)

    original, baked = load_file(path), load_file(out)
    for key, tensor in original.items():
        assert torch.equal(tensor, baked[key]), key


def test_a_slider_at_one_still_folds_a_non_unit_scale_in(tmp_path, flux_lora):
    # alpha=2, rank=4, so the source runs at scale 0.5; the baked file has to keep contributing
    # 0.5x even though its own scale is now 1.0
    primary, _donor, baked, _summary = bake_with(tmp_path, flux_lora, {})

    for name in FLUX_MODULES:
        want = contribution(primary.modules[name], 1.0)
        assert torch.allclose(want, contribution(baked.modules[name], 1.0), atol=1e-5), name


# --- donor blending --------------------------------------------------------------------------------


def test_a_blended_block_is_the_sum_of_both_contributions(tmp_path, flux_lora, donor_lora):
    primary, donor, baked, _summary = bake_with(
        tmp_path, flux_lora, {"double_0": 0.4}, donor_lora, {"double_0": 0.6})

    for name in primary.modules_in("double_0"):
        want = contribution(primary.modules[name], 0.4) + contribution(donor.modules[name], 0.6)
        got = contribution(baked.modules[name], 1.0)
        # rank concatenation is exact: no SVD, no approximation, just a wider pair of matrices
        assert torch.allclose(want, got, atol=1e-5), name


def test_blending_widens_the_rank_by_exactly_the_donors(tmp_path, flux_lora, donor_lora):
    primary, donor, baked, summary = bake_with(
        tmp_path, flux_lora, {"double_0": 0.4}, donor_lora, {"double_0": 0.6})

    name = primary.modules_in("double_0")[0]
    assert baked.modules[name].rank == primary.modules[name].rank + donor.modules[name].rank
    assert summary.blended == ["double_0"]


def test_a_donor_at_zero_costs_nothing(tmp_path, flux_lora, donor_lora):
    primary, _donor, baked, summary = bake_with(tmp_path, flux_lora, {}, donor_lora)

    # merely loading a donor must not double every block with a half that contributes nothing
    name = FLUX_MODULES[0]
    assert baked.modules[name].rank == primary.modules[name].rank
    assert summary.blended == []


def test_balancing_holds_the_block_total_at_one(tmp_path, flux_lora, donor_lora):
    primary = lora_file.load(flux_lora)
    donor = lora_file.load(donor_lora)
    state = SliderState.for_blocks(b.id for b in primary.blocks)
    state.blocks["double_0"].primary_strength = 0.25
    state.balance("double_0")

    assert state.blocks["double_0"].donor_strength == 0.75

    out = str(tmp_path / "baked.safetensors")
    bake_module.bake(primary, state, out, donor)
    baked = lora_file.load(out)

    name = primary.modules_in("double_0")[0]
    want = (contribution(primary.modules[name], 0.25) + contribution(donor.modules[name], 0.75))
    assert torch.allclose(want, contribution(baked.modules[name], 1.0), atol=1e-5)


def test_a_donor_from_a_different_model_is_refused(tmp_path, flux_lora):
    mismatched = write_lora(tmp_path / "other.safetensors", FLUX_MODULES, out=16)
    primary = lora_file.load(flux_lora)
    donor = lora_file.load(mismatched)
    state = SliderState.for_blocks(b.id for b in primary.blocks)
    state.blocks["double_0"].donor_strength = 0.5

    with pytest.raises(UnsupportedLoRA, match="different models"):
        bake_module.bake(primary, state, str(tmp_path / "out.safetensors"), donor)


# --- what travels with the file ---------------------------------------------------------------------


def test_the_source_content_hash_is_not_inherited(tmp_path):
    path = write_lora(tmp_path / "hashed.safetensors", FLUX_MODULES,
                      metadata={"sshs_model_hash": "deadbeef", "ss_base_model_version": "flux1"})
    primary = lora_file.load(path)
    state = SliderState.for_blocks(b.id for b in primary.blocks)

    out = str(tmp_path / "baked.safetensors")
    bake_module.bake(primary, state, out, None)
    metadata = lora_file.read_metadata(out)

    assert "sshs_model_hash" not in metadata   # it described the file this one came from
    assert metadata["ss_base_model_version"] == "flux1"   # this still describes it


def test_the_slider_settings_are_recorded_in_the_file(tmp_path, flux_lora):
    primary = lora_file.load(flux_lora)
    state = SliderState.for_blocks(b.id for b in primary.blocks)
    state.blocks["double_0"].primary_strength = 0.4

    out = str(tmp_path / "baked.safetensors")
    bake_module.bake(primary, state, out, None)

    recorded = json.loads(lora_file.read_metadata(out)["ot_repair_studio"])
    assert recorded["blocks"]["double_0"]["primary_strength"] == 0.4


def test_the_reported_rank_matches_the_file(tmp_path, flux_lora, donor_lora):
    _primary, _donor, _baked, summary = bake_with(
        tmp_path, flux_lora, {"double_0": 0.4}, donor_lora, {"double_0": 0.6})

    metadata = lora_file.read_metadata(summary.out_path)
    assert int(metadata["ss_network_dim"]) == summary.max_rank == 6  # 4 primary + 2 donor


def test_bundled_embeddings_survive_a_bake(tmp_path):
    path = tmp_path / "bundled.safetensors"
    generator = torch.Generator().manual_seed(3)
    state_dict = {
        "lora_unet_double_blocks_0_img_attn_proj.lora_down.weight": torch.randn(4, 6, generator=generator),
        "lora_unet_double_blocks_0_img_attn_proj.lora_up.weight": torch.randn(8, 4, generator=generator),
        "bundle_emb.myembed.clip_l": torch.randn(2, 768, generator=generator),
    }
    save_file(state_dict, str(path))

    primary = lora_file.load(str(path))
    out = str(tmp_path / "baked.safetensors")
    bake_module.bake(primary, SliderState.for_blocks(b.id for b in primary.blocks), out, None)

    assert torch.equal(load_file(out)["bundle_emb.myembed.clip_l"], state_dict["bundle_emb.myembed.clip_l"])


# --- the slider state -------------------------------------------------------------------------------


def test_the_quick_sets_do_what_they_say():
    state = SliderState.for_blocks(["a"])

    state.zero("a")
    assert state.blocks["a"].primary_strength == 0.0
    assert not state.blocks["a"].contributes_primary()

    state.full("a")
    assert state.blocks["a"].primary_strength == 1.0

    state.invert("a")
    assert state.blocks["a"].primary_strength == -1.0


def test_a_slider_cannot_be_dragged_past_the_point_of_meaning():
    state = SliderState.for_blocks(["a"])

    state.set_strength("a", 99.0)
    assert state.blocks["a"].primary_strength == 3.0

    state.set_strength("a", -99.0)
    assert state.blocks["a"].primary_strength == -3.0


def test_balance_works_from_either_side():
    state = SliderState.for_blocks(["a"])

    state.set_strength("a", 0.3)
    state.balance("a")
    assert state.blocks["a"].donor_strength == pytest.approx(0.7)

    state.set_strength("a", 0.9, donor=True)
    state.balance("a", from_donor=True)
    assert state.blocks["a"].primary_strength == pytest.approx(0.1)


def test_a_quick_set_can_be_applied_to_everything():
    state = SliderState.for_blocks(["a", "b", "c"])

    state.apply_all("zero")

    assert all(not s.contributes_primary() for s in state.blocks.values())


def test_switching_to_a_new_lora_keeps_the_edits_that_still_apply():
    state = SliderState.for_blocks(["double_0", "double_1"])
    state.set_strength("double_0", 0.4)

    resized = state.resize_to(["double_0", "single_0"])

    assert resized.blocks["double_0"].primary_strength == 0.4   # same block, kept
    assert resized.blocks["single_0"].primary_strength == 1.0   # new block, default
    assert "double_1" not in resized.blocks                     # gone with the old LoRA


def test_a_preset_survives_the_round_trip(tmp_path):
    state = SliderState.for_blocks(["double_0", "single_0"])
    state.set_strength("double_0", -0.5)
    state.set_strength("single_0", 0.75, donor=True)
    path = str(tmp_path / "preset.json")

    state.save(path)

    assert SliderState.load(path).blocks == state.blocks


def test_a_preset_written_by_a_later_version_still_loads():
    # an unknown field must not stop the preset opening
    loaded = SliderState.from_json({"blocks": {"a": {"primary_strength": 0.5, "future_thing": 7}}})

    assert loaded.blocks["a"].primary_strength == 0.5


def test_the_default_state_would_change_nothing():
    assert SliderState.for_blocks(["a", "b"]).is_default()


# --- the controller ------------------------------------------------------------------------------


def test_opening_a_lora_builds_a_slider_for_every_block(flux_lora):
    controller = RepairStudioTabController()

    assert controller.load_primary(flux_lora) is None
    assert set(controller.state.blocks) == {"double_0", "double_1", "single_0", "single_11"}


def test_the_sliders_are_grouped_the_way_the_model_runs(flux_lora):
    controller = RepairStudioTabController()
    controller.load_primary(flux_lora)

    headings = [heading for heading, _blocks in controller.grouped_blocks()]
    assert headings == ["Double blocks", "Single blocks"]


def test_browsing_a_second_lora_keeps_the_edits(tmp_path, flux_lora):
    controller = RepairStudioTabController()
    controller.load_primary(flux_lora)
    controller.set_strength("double_0", 0.25)

    # the next checkpoint of the same run
    second = write_lora(tmp_path / "epoch2.safetensors", FLUX_MODULES, seed=9)
    controller.load_primary(second)

    assert controller.state.blocks["double_0"].primary_strength == 0.25


def test_a_lora_that_cannot_be_opened_reports_why(tmp_path):
    # LoKR/LoHa open and edit exactly now; Tucker is the variant that still cannot
    path = tmp_path / "tucker.safetensors"
    save_file({"lora_unet_blocks_0_x.lokr_w1": torch.randn(2, 2),
               "lora_unet_blocks_0_x.lokr_w2": torch.randn(2, 2),
               "lora_unet_blocks_0_x.lokr_t2": torch.randn(2, 2, 2)}, str(path))
    controller = RepairStudioTabController()

    error = controller.load_primary(str(path))

    assert error is not None
    assert "Tucker" in error


def test_a_donor_with_nothing_in_common_is_rejected_on_load(tmp_path, flux_lora):
    other = write_lora(tmp_path / "sdxl.safetensors",
                       ["lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q"])
    controller = RepairStudioTabController()
    controller.load_primary(flux_lora)

    error = controller.load_donor(other)

    assert error is not None
    assert "different models" in error
    assert not controller.has_donor   # and it is not left half-attached


def test_dragging_a_slider_off_zero_brings_the_block_back(flux_lora):
    controller = RepairStudioTabController()
    controller.load_primary(flux_lora)
    controller.apply("zero", "double_0")

    controller.set_strength("double_0", 0.5)

    assert controller.state.blocks["double_0"].contributes_primary()


def test_the_summary_says_what_the_bake_would_do(flux_lora):
    controller = RepairStudioTabController()
    controller.load_primary(flux_lora)
    controller.apply("zero", "double_0")
    controller.set_strength("single_0", 0.5)

    summary = controller.bake_summary()

    assert "1 block(s) removed" in summary
    assert "rescaled" in summary


def test_saving_over_the_lora_being_edited_is_refused(flux_lora):
    controller = RepairStudioTabController()
    controller.load_primary(flux_lora)

    error, _message = controller.save(flux_lora)

    assert error is not None
    assert "overwrite" in error


def test_the_suggested_filename_sits_beside_the_original(flux_lora):
    controller = RepairStudioTabController()
    controller.load_primary(flux_lora)

    assert controller.default_output_path().endswith("primary-repaired.safetensors")


def test_a_preset_from_another_model_does_not_add_phantom_sliders(tmp_path, flux_lora):
    other = SliderState.for_blocks(["in_0", "in_1"])
    other.set_strength("in_0", 0.2)
    preset = str(tmp_path / "sdxl.json")
    other.save(preset)

    controller = RepairStudioTabController()
    controller.load_primary(flux_lora)
    assert controller.load_preset(preset) is None

    assert set(controller.state.blocks) == {"double_0", "double_1", "single_0", "single_11"}


def test_saving_reports_what_it_wrote(tmp_path, flux_lora):
    controller = RepairStudioTabController()
    controller.load_primary(flux_lora)
    controller.apply("zero", "double_0")

    error, message = controller.save(str(tmp_path / "out.safetensors"))

    assert error is None
    assert "Saved out.safetensors" in message
    assert os.path.exists(tmp_path / "out.safetensors")


def test_the_sgm_middle_block_is_one_slider_not_three():
    # middle_block.0 / .1 / .2 are the resblock, attention and resblock *inside* one middle block
    ids = {block_for(f"lora_unet_middle_block_{i}_x").id for i in range(3)}

    assert ids == {"mid_0"}


# --- the tab, in both UI frameworks ----------------------------------------------------------------


@pytest.fixture(scope="module")
def ctk_root():
    ctk = pytest.importorskip("customtkinter")
    try:
        root = ctk.CTk()
    except Exception as e:  # no display
        pytest.skip(f"no Tk display available: {e}")
    root.withdraw()
    yield root
    root.destroy()


def build_ctk_tab(root):
    from modules.ui.CtkRepairStudioTabView import CtkRepairStudioTabView

    import customtkinter as ctk

    frame = ctk.CTkFrame(root)
    return CtkRepairStudioTabView(frame, RepairStudioTabController())


def ctk_texts(widget) -> list[str]:
    import customtkinter as ctk

    texts = []
    for child in widget.winfo_children():
        if isinstance(child, ctk.CTkLabel):
            text = child.cget("text")
            if text:
                texts.append(text)
        else:
            texts.extend(ctk_texts(child))
    return texts


def test_ctk_the_tab_opens_with_nothing_loaded(ctk_root):
    tab = build_ctk_tab(ctk_root)

    assert tab.rows == {}
    assert "No LoRA open." in ctk_texts(tab.scroll_frame)


def test_ctk_opening_a_lora_builds_one_row_per_block(ctk_root, flux_lora):
    tab = build_ctk_tab(ctk_root)

    tab.load_primary(flux_lora)

    assert set(tab.rows) == {"double_0", "double_1", "single_0", "single_11"}
    texts = ctk_texts(tab.scroll_frame)
    assert "Double blocks" in texts
    assert any(t.startswith("Single block 11") for t in texts)


def test_ctk_a_quick_set_moves_the_slider_it_belongs_to(ctk_root, flux_lora):
    tab = build_ctk_tab(ctk_root)
    tab.load_primary(flux_lora)

    tab.controller.apply("zero", "double_0")
    tab._CtkRepairStudioTabView__sync_row("double_0")

    assert tab.rows["double_0"]["primary_slider"].get() == 0.0
    assert tab.rows["double_0"]["primary_value"].cget("text") == "+0.00"


def test_ctk_the_donor_sliders_appear_only_with_a_donor(ctk_root, flux_lora, donor_lora):
    tab = build_ctk_tab(ctk_root)
    tab.load_primary(flux_lora)
    assert "donor_slider" not in tab.rows["double_0"]

    tab.load_donor(donor_lora)

    assert "donor_slider" in tab.rows["double_0"]


def test_ctk_a_rejected_donor_leaves_no_donor_sliders(ctk_root, tmp_path, flux_lora):
    tab = build_ctk_tab(ctk_root)
    tab.load_primary(flux_lora)
    other = write_lora(tmp_path / "sdxl.safetensors", ["lora_unet_input_blocks_4_1_attn"])

    tab.load_donor(other)

    assert "donor_slider" not in tab.rows["double_0"]
    assert "different models" in tab.status_label.cget("text")


def test_ctk_the_summary_updates_as_sliders_move(ctk_root, flux_lora):
    tab = build_ctk_tab(ctk_root)
    tab.load_primary(flux_lora)

    tab._CtkRepairStudioTabView__quick_set("double_0", "zero", donor=False)

    assert "removed" in tab.summary_label.cget("text")


def test_ctk_switching_lora_rebuilds_the_rows(ctk_root, tmp_path, flux_lora):
    tab = build_ctk_tab(ctk_root)
    tab.load_primary(flux_lora)

    sdxl = write_lora(tmp_path / "sdxl.safetensors", [
        "lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q",
        "lora_unet_output_blocks_5_1_transformer_blocks_0_attn1_to_q",
    ])
    tab.load_primary(sdxl)

    assert set(tab.rows) == {"in_4", "out_5"}


@pytest.fixture(scope="module")
def qt_app():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def build_qt_tab():
    from modules.ui.PySide6RepairStudioTabView import PySide6RepairStudioTabView

    return PySide6RepairStudioTabView(None, RepairStudioTabController())


def qt_texts(widget) -> list[str]:
    from PySide6.QtWidgets import QLabel

    return [child.text() for child in widget.findChildren(QLabel) if child.text()]


def test_qt_the_tab_opens_with_nothing_loaded(qt_app):
    tab = build_qt_tab()

    assert tab.rows == {}
    assert "No LoRA open." in qt_texts(tab)


def test_qt_opening_a_lora_builds_one_row_per_block(qt_app, flux_lora):
    tab = build_qt_tab()

    tab.load_primary(flux_lora)

    assert set(tab.rows) == {"double_0", "double_1", "single_0", "single_11"}
    assert "Double blocks" in qt_texts(tab)


def test_qt_a_quick_set_moves_the_slider_it_belongs_to(qt_app, flux_lora):
    tab = build_qt_tab()
    tab.load_primary(flux_lora)

    tab._PySide6RepairStudioTabView__quick_set("double_0", "zero", donor=False)

    assert tab.rows["double_0"]["primary_slider"].value() == 0
    assert tab.rows["double_0"]["primary_value"].text() == "+0.00"


def test_qt_syncing_a_row_does_not_write_the_rounded_value_back(qt_app, flux_lora):
    tab = build_qt_tab()
    tab.load_primary(flux_lora)

    # QSlider is integer-only; without blockSignals the setValue below would fire valueChanged and
    # overwrite the state with the rounded number
    tab.controller.set_strength("double_0", 0.333)
    tab._PySide6RepairStudioTabView__sync_row("double_0")

    assert tab.controller.state.blocks["double_0"].primary_strength == pytest.approx(0.333)


def test_qt_the_donor_sliders_appear_only_with_a_donor(qt_app, flux_lora, donor_lora):
    tab = build_qt_tab()
    tab.load_primary(flux_lora)
    assert "donor_slider" not in tab.rows["double_0"]

    tab.load_donor(donor_lora)

    assert "donor_slider" in tab.rows["double_0"]


def test_qt_switching_lora_leaves_no_stale_rows(qt_app, tmp_path, flux_lora):
    tab = build_qt_tab()
    tab.load_primary(flux_lora)

    sdxl = write_lora(tmp_path / "sdxl.safetensors",
                      ["lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q"])
    tab.load_primary(sdxl)

    assert set(tab.rows) == {"in_4"}
    assert not any(t.startswith("Double block") for t in qt_texts(tab))


# --- LyCORIS, edited losslessly -------------------------------------------------------------------
#
# LoKR and LoHa deltas are linear in their first factor, so a per-block multiplier folds into that
# factor exactly — no SVD, no format conversion, alpha untouched. Every claim here is judged by
# densifying the delta and measuring, because "lossless" is the kind of word that needs evidence.


LOKR_MODULES = ["lora_unet_double_blocks_0_img_attn_proj", "lora_unet_single_blocks_2_linear1"]


def write_lokr(path, module_names, *, factored=False, alpha=2.0, seed=0):
    generator = torch.Generator().manual_seed(seed)
    state_dict = {}
    for name in module_names:
        if factored:
            state_dict[f"{name}.lokr_w1_a"] = torch.randn(4, 2, generator=generator)
            state_dict[f"{name}.lokr_w1_b"] = torch.randn(2, 3, generator=generator)
        else:
            state_dict[f"{name}.lokr_w1"] = torch.randn(4, 3, generator=generator)
        state_dict[f"{name}.lokr_w2"] = torch.randn(2, 2, generator=generator)
        state_dict[f"{name}.alpha"] = torch.tensor(float(alpha))
    save_file(state_dict, str(path))
    return str(path)


def write_loha(path, module_names, *, alpha=2.0, rank=2, seed=0):
    generator = torch.Generator().manual_seed(seed)
    state_dict = {}
    for name in module_names:
        for i in (1, 2):
            state_dict[f"{name}.hada_w{i}_a"] = torch.randn(8, rank, generator=generator)
            state_dict[f"{name}.hada_w{i}_b"] = torch.randn(rank, 6, generator=generator)
        state_dict[f"{name}.alpha"] = torch.tensor(float(alpha))
    save_file(state_dict, str(path))
    return str(path)


def bake_lycoris(tmp_path, source_path, edits: dict):
    primary = lora_file.load(source_path)
    state = SliderState.for_blocks(b.id for b in primary.blocks)
    for block_id, value in edits.items():
        state.blocks[block_id].primary_strength = value
    out = str(tmp_path / "baked.safetensors")
    summary = bake_module.bake(primary, state, out, None)
    return primary, lora_file.load(out), summary


@pytest.mark.parametrize("factored", [False, True], ids=["full-matrix", "factored"])
def test_a_rescaled_lokr_block_is_exactly_the_slider_times_the_original(tmp_path, factored):
    source = write_lokr(tmp_path / "lokr.safetensors", LOKR_MODULES, factored=factored, seed=1)

    primary, baked, _summary = bake_lycoris(tmp_path, source, {"double_0": 0.35})

    for name in primary.modules_in("double_0"):
        want = contribution(primary.modules[name], 0.35)
        got = contribution(baked.modules[name], 1.0)
        assert torch.allclose(want, got, atol=1e-5), name


def test_a_rescaled_loha_block_is_exactly_the_slider_times_the_original(tmp_path):
    source = write_loha(tmp_path / "loha.safetensors", LOKR_MODULES, seed=2)

    primary, baked, _summary = bake_lycoris(tmp_path, source, {"single_2": -1.5})

    for name in primary.modules_in("single_2"):
        want = contribution(primary.modules[name], -1.5)
        assert torch.allclose(want, contribution(baked.modules[name], 1.0), atol=1e-5), name


def test_an_untouched_lycoris_module_is_byte_identical(tmp_path):
    # Fizgig's headline claim for its own lossless bake, held here too: a saved file with no edits
    # IS the original, tensor for tensor — alpha included
    source = write_lokr(tmp_path / "lokr.safetensors", LOKR_MODULES, seed=3)

    _primary, _baked, _summary = bake_lycoris(tmp_path, source, {})

    original, baked = load_file(source), load_file(str(tmp_path / "baked.safetensors"))
    assert set(original) == set(baked)
    for key in original:
        assert torch.equal(original[key], baked[key]), key


def test_only_the_first_factor_changes(tmp_path):
    source = write_lokr(tmp_path / "lokr.safetensors", LOKR_MODULES, factored=True, seed=4)

    _primary, _baked, _summary = bake_lycoris(tmp_path, source, {"double_0": 0.5})

    original, baked = load_file(source), load_file(str(tmp_path / "baked.safetensors"))
    name = LOKR_MODULES[0]
    assert not torch.equal(original[f"{name}.lokr_w1_a"], baked[f"{name}.lokr_w1_a"])
    # everything else in the module — including the second factor and alpha — is untouched, which
    # is what keeps the file loadable by anything that loaded the original
    for suffix in ("lokr_w1_b", "lokr_w2", "alpha"):
        assert torch.equal(original[f"{name}.{suffix}"], baked[f"{name}.{suffix}"]), suffix


def test_the_lycoris_alpha_is_never_rewritten(tmp_path):
    # the standard bake normalises alpha = rank; the LyCORIS bake must NOT do the analogue, because
    # the sentinel-alpha convention it would need is not read by every loader
    source = write_lokr(tmp_path / "lokr.safetensors", LOKR_MODULES, alpha=2.0, seed=5)

    _primary, baked, _summary = bake_lycoris(tmp_path, source, {"double_0": 0.5})

    assert baked.modules[LOKR_MODULES[0]].alpha == 2.0


def test_a_zeroed_lycoris_block_is_gone_from_the_file(tmp_path):
    source = write_lokr(tmp_path / "lokr.safetensors", LOKR_MODULES, seed=6)
    primary = lora_file.load(source)
    state = SliderState.for_blocks(b.id for b in primary.blocks)
    state.zero("double_0")

    out = str(tmp_path / "baked.safetensors")
    summary = bake_module.bake(primary, state, out, None)

    assert summary.dropped == ["double_0"]
    assert not any(block_for(name).id == "double_0" for name in lora_file.load(out).modules)


def test_the_lokr_scale_convention_matches_lycoris_inference(tmp_path):
    # alpha / min(active decomposition ranks); a full-matrix LoKR has dim 1, so scale = alpha.
    # Getting this wrong silently rescales every delta the tests then "verify" against itself,
    # which is why it is pinned explicitly.
    full = lora_file.load(write_lokr(tmp_path / "full.safetensors", LOKR_MODULES, alpha=2.0))
    factored = lora_file.load(write_lokr(tmp_path / "fact.safetensors", LOKR_MODULES,
                                         factored=True, alpha=2.0))

    assert full.modules[LOKR_MODULES[0]].scale == 2.0       # alpha / 1
    assert factored.modules[LOKR_MODULES[0]].scale == 1.0   # alpha / rank 2


def test_a_sentinel_alpha_means_the_scale_is_already_baked(tmp_path):
    path = tmp_path / "refined.safetensors"
    save_file({
        f"{LOKR_MODULES[0]}.lokr_w1": torch.randn(4, 3),
        f"{LOKR_MODULES[0]}.lokr_w2": torch.randn(2, 2),
        f"{LOKR_MODULES[0]}.alpha": torch.tensor(1e10),
    }, str(path))

    # Fizgig's refined exports write alpha = 1e10 to mean scale 1.0; reading it as alpha/dim
    # would multiply the delta by ten billion
    assert lora_file.load(str(path)).modules[LOKR_MODULES[0]].scale == 1.0


def test_blending_a_lycoris_block_is_refused_not_approximated(tmp_path, donor_lora):
    source = write_lokr(tmp_path / "lokr.safetensors", FLUX_MODULES, seed=7)
    primary = lora_file.load(source)
    donor = lora_file.load(donor_lora)
    state = SliderState.for_blocks(b.id for b in primary.blocks)
    state.blocks["double_0"].donor_strength = 0.5

    with pytest.raises(UnsupportedLoRA, match="exactly"):
        bake_module.bake(primary, state, str(tmp_path / "out.safetensors"), donor)


def test_a_tucker_lokr_is_refused_with_a_reason(tmp_path):
    path = tmp_path / "tucker.safetensors"
    save_file({
        f"{LOKR_MODULES[0]}.lokr_w1": torch.randn(4, 3),
        f"{LOKR_MODULES[0]}.lokr_w2": torch.randn(2, 2),
        f"{LOKR_MODULES[0]}.lokr_t2": torch.randn(2, 2, 2),
    }, str(path))

    with pytest.raises(UnsupportedLoRA, match="Tucker"):
        lora_file.load(str(path))


def test_a_lokr_missing_its_second_half_is_caught(tmp_path):
    path = tmp_path / "half.safetensors"
    save_file({f"{LOKR_MODULES[0]}.lokr_w1": torch.randn(4, 3)}, str(path))

    with pytest.raises(UnsupportedLoRA, match="truncated"):
        lora_file.load(str(path))


def test_a_mixed_file_edits_both_forms_correctly(tmp_path):
    # one file, one block standard and one LoKR — real exports mix forms
    generator = torch.Generator().manual_seed(8)
    path = tmp_path / "mixed.safetensors"
    save_file({
        "lora_unet_double_blocks_0_img_attn_proj.lora_down.weight": torch.randn(4, 6, generator=generator),
        "lora_unet_double_blocks_0_img_attn_proj.lora_up.weight": torch.randn(8, 4, generator=generator),
        "lora_unet_double_blocks_0_img_attn_proj.alpha": torch.tensor(2.0),
        "lora_unet_single_blocks_2_linear1.lokr_w1": torch.randn(4, 3, generator=generator),
        "lora_unet_single_blocks_2_linear1.lokr_w2": torch.randn(2, 2, generator=generator),
        "lora_unet_single_blocks_2_linear1.alpha": torch.tensor(2.0),
    }, str(path))

    primary, baked, _summary = bake_lycoris(tmp_path, str(path), {"double_0": 0.5, "single_2": 0.25})

    for name, multiplier in (("lora_unet_double_blocks_0_img_attn_proj", 0.5),
                             ("lora_unet_single_blocks_2_linear1", 0.25)):
        want = contribution(primary.modules[name], multiplier)
        assert torch.allclose(want, contribution(baked.modules[name], 1.0), atol=1e-5), name


def test_the_controller_opens_a_lokr_and_says_so(tmp_path):
    source = write_lokr(tmp_path / "lokr.safetensors", LOKR_MODULES, seed=9)
    controller = RepairStudioTabController()

    assert controller.load_primary(source) is None
    assert "LoKR" in controller.primary_summary()
    assert set(controller.state.blocks) == {"double_0", "single_2"}


def test_a_lycoris_donor_pair_does_not_crash_the_mismatch_check(tmp_path):
    primary_path = write_lokr(tmp_path / "p.safetensors", LOKR_MODULES, seed=10)
    donor_path = write_lokr(tmp_path / "d.safetensors", LOKR_MODULES, seed=11)
    controller = RepairStudioTabController()
    controller.load_primary(primary_path)

    # loads fine; the exactness refusal only happens if a blend is actually asked for
    assert controller.load_donor(donor_path) is None


# --- DoRA: partial strengths refused, not silently wrong ------------------------------------------


def write_dora(path, module_names, *, seed=0):
    generator = torch.Generator().manual_seed(seed)
    state_dict = {}
    for name in module_names:
        state_dict[f"{name}.lora_down.weight"] = torch.randn(4, 6, generator=generator)
        state_dict[f"{name}.lora_up.weight"] = torch.randn(8, 4, generator=generator)
        state_dict[f"{name}.alpha"] = torch.tensor(4.0)
        state_dict[f"{name}.dora_scale"] = torch.randn(8, 1, generator=generator)
    save_file(state_dict, str(path))
    return str(path)


def test_rescaling_a_dora_block_is_refused_with_the_reason(tmp_path):
    # a DoRA delta is renormalised at load time, so scaling its matrices does not scale its effect;
    # the old behaviour was worse than refusal — the dora_scale key was silently dropped
    source = write_dora(tmp_path / "dora.safetensors", FLUX_MODULES, seed=12)
    primary = lora_file.load(source)
    state = SliderState.for_blocks(b.id for b in primary.blocks)
    state.blocks["double_0"].primary_strength = 0.5

    with pytest.raises(UnsupportedLoRA, match="DoRA"):
        bake_module.bake(primary, state, str(tmp_path / "out.safetensors"), None)


def test_an_untouched_dora_block_passes_through_whatever_its_alpha(tmp_path):
    # a DoRA module at strength 1.0 must come out byte-identical even when alpha != rank —
    # normalising its scale would be the very edit the DoRA guard exists to refuse
    generator = torch.Generator().manual_seed(14)
    path = tmp_path / "dora_odd_alpha.safetensors"
    name = FLUX_MODULES[0]
    save_file({
        f"{name}.lora_down.weight": torch.randn(4, 6, generator=generator),
        f"{name}.lora_up.weight": torch.randn(8, 4, generator=generator),
        f"{name}.alpha": torch.tensor(2.0),  # scale 0.5, deliberately not 1.0
        f"{name}.dora_scale": torch.randn(8, 1, generator=generator),
    }, str(path))
    primary = lora_file.load(str(path))
    state = SliderState.for_blocks(b.id for b in primary.blocks)

    out = str(tmp_path / "baked.safetensors")
    bake_module.bake(primary, state, out, None)

    original, baked = load_file(str(path)), load_file(out)
    for key in original:
        assert torch.equal(original[key], baked[key]), key


def test_a_dora_block_at_full_strength_keeps_its_dora_scale(tmp_path):
    source = write_dora(tmp_path / "dora.safetensors", FLUX_MODULES, seed=13)
    primary = lora_file.load(source)
    state = SliderState.for_blocks(b.id for b in primary.blocks)
    state.zero("single_0")  # dropping is fine — the whole block goes, dora key included

    out = str(tmp_path / "baked.safetensors")
    bake_module.bake(primary, state, out, None)

    baked = load_file(out)
    name = FLUX_MODULES[0]  # an untouched block
    assert torch.equal(load_file(source)[f"{name}.dora_scale"], baked[f"{name}.dora_scale"])
    assert not any(k.startswith("lora_unet_single_blocks_0_") for k in baked)
