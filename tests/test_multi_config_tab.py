"""Builds the Multi Config tab for real and drives its dropdowns.

The tab rebuilds parts of itself when the mode, the swept setting or the value count changes, which
is the kind of thing that only breaks once it is actually constructed. These tests instantiate the
real views in both UI frameworks and check what ends up on screen and in the config.

Run with:  pytest tests/test_multi_config_tab.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.ui.MultiConfigTabController import MultiConfigTabController
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.MultiConfigMode import MultiConfigMode
from modules.util.enum.MultiConfigSweepSetting import MultiConfigSweepSetting

import pytest


def make_config() -> TrainConfig:
    config = TrainConfig.default_values()
    config.multi_config = True
    config.multi_config_sweep_count = 3
    return config


# --- controller (no widgets) --------------------------------------------------------------------


def test_every_sweep_setting_is_offered():
    controller = MultiConfigTabController(make_config())
    offered = {setting for _label, setting in controller.get_sweep_settings()}
    assert offered == set(MultiConfigSweepSetting)


def test_value_counts_run_from_one_to_ten():
    controller = MultiConfigTabController(make_config())
    assert controller.get_sweep_counts() == [str(i) for i in range(1, 11)]


def test_choice_settings_offer_choices_and_learning_rate_does_not():
    controller = MultiConfigTabController(make_config())

    controller.config.multi_config_sweep_setting = MultiConfigSweepSetting.OPTIMIZER
    assert controller.sweep_is_choice()
    assert "ADAMW" in controller.sweep_choices()

    controller.config.multi_config_sweep_setting = MultiConfigSweepSetting.LEARNING_RATE
    assert not controller.sweep_is_choice()
    assert controller.sweep_choices() == []


def test_only_layer_filter_warns_about_independent_lineages():
    controller = MultiConfigTabController(make_config())

    controller.config.multi_config_sweep_setting = MultiConfigSweepSetting.LEARNING_RATE
    assert controller.sweep_notice() is None

    controller.config.multi_config_sweep_setting = MultiConfigSweepSetting.LAYER_FILTER
    assert "own model" in controller.sweep_notice()


def test_the_adaptive_mode_is_offered():
    controller = MultiConfigTabController(make_config())
    offered = {mode for _label, mode in controller.get_modes()}
    assert offered == set(MultiConfigMode)


def test_the_ladder_preview_shows_the_first_round():
    config = make_config()
    config.multi_config_adaptive_lr = 0.0003
    controller = MultiConfigTabController(config)

    assert controller.ladder_preview() == "0.0002 / 0.0003 / 0.0004"


def test_the_ladder_preview_says_when_a_value_was_snapped():
    config = make_config()
    config.multi_config_adaptive_lr = 0.00034
    controller = MultiConfigTabController(config)

    preview = controller.ladder_preview()
    assert preview.startswith("0.0002 / 0.0003 / 0.0004")
    assert "snapped from 0.00034" in preview


@pytest.mark.parametrize("value", [0.0, -1e-4])
def test_the_ladder_preview_copes_with_an_unusable_value(value):
    # recomputed on every keystroke, so it has to survive half-typed numbers
    config = make_config()
    config.multi_config_adaptive_lr = value
    controller = MultiConfigTabController(config)

    assert "greater than zero" in controller.ladder_preview()


def test_values_that_no_longer_fit_the_setting_are_repaired():
    config = make_config()
    config.multi_config_sweep_setting = MultiConfigSweepSetting.LEARNING_RATE
    config.multi_config_sweep_value_1 = "1e-4"
    config.multi_config_sweep_value_2 = "1e-3"
    config.multi_config_sweep_value_3 = "1e-2"
    controller = MultiConfigTabController(config)

    assert controller.repair_sweep_values() == {}, "valid learning rates should be left alone"

    config.multi_config_sweep_setting = MultiConfigSweepSetting.OPTIMIZER
    repaired = controller.repair_sweep_values()

    # numbers are not optimizer names, so every slot is replaced with something valid
    assert "multi_config_sweep_value_1" in repaired
    assert repaired["multi_config_sweep_value_1"] in controller.sweep_choices()


def test_valid_values_survive_switching_settings_and_back():
    config = make_config()
    config.multi_config_sweep_setting = MultiConfigSweepSetting.OPTIMIZER
    config.multi_config_sweep_value_1 = "PRODIGY"
    controller = MultiConfigTabController(config)

    repaired = controller.repair_sweep_values()
    assert "multi_config_sweep_value_1" not in repaired


# --- CustomTkinter view -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ctk_root():
    # one root for the whole module: each tab is built into its own frame, and repeatedly building
    # and tearing down Tk roots in a single process is what makes these suites flaky
    ctk = pytest.importorskip("customtkinter")
    try:
        root = ctk.CTk()
    except Exception as e:  # no display
        pytest.skip(f"no Tk display available: {e}")
    root.withdraw()
    yield root
    root.destroy()


def build_ctk_tab(root, config):
    from modules.ui.CtkMultiConfigTabView import CtkMultiConfigTabView
    from modules.util.ui.CtkUIState import CtkUIState

    import customtkinter as ctk

    frame = ctk.CTkFrame(root)
    ui_state = CtkUIState(root, config)
    return CtkMultiConfigTabView(frame, MultiConfigTabController(config), ui_state)


def ctk_texts(widget) -> list[str]:
    # only CTkLabel: a CTkLabel is a frame wrapping a plain tk Label, so walking every widget with a
    # "text" option would report each label twice
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


def test_ctk_tab_starts_in_full_config_mode(ctk_root):
    config = make_config()
    tab = build_ctk_tab(ctk_root, config)

    texts = ctk_texts(tab.scroll_frame)
    assert "Number of Configs" in texts
    for slot in range(1, config.multi_config_path_count + 1):
        assert f"Config Set {slot}" in texts
    assert f"Config Set {config.multi_config_path_count + 1}" not in texts
    assert "Selection Metric" in texts


def test_ctk_tab_switches_to_the_sweep_inputs(ctk_root):
    config = make_config()
    config.multi_config_sweep_setting = MultiConfigSweepSetting.LEARNING_RATE
    tab = build_ctk_tab(ctk_root, config)

    tab.ui_state.get_var("multi_config_mode").set(str(MultiConfigMode.SINGLE_SETTING))
    tab.refresh_mode_frame()

    texts = ctk_texts(tab.scroll_frame)
    assert "Setting" in texts
    assert "Number of Values" in texts
    assert "Config Set 1" not in texts
    # one labelled box per value, matching the count
    assert [t for t in texts if t.startswith("Learning Rate ")] == [
        "Learning Rate 1", "Learning Rate 2", "Learning Rate 3"
    ]


def test_ctk_value_count_changes_the_number_of_inputs(ctk_root):
    config = make_config()
    config.multi_config_mode = MultiConfigMode.SINGLE_SETTING
    config.multi_config_sweep_setting = MultiConfigSweepSetting.LEARNING_RATE
    tab = build_ctk_tab(ctk_root, config)

    tab.ui_state.get_var("multi_config_sweep_count").set("7")
    tab.refresh_values_frame()

    texts = ctk_texts(tab.values_frame)
    assert len([t for t in texts if t.startswith("Learning Rate ")]) == 7

    tab.ui_state.get_var("multi_config_sweep_count").set("2")
    tab.refresh_values_frame()

    texts = ctk_texts(tab.values_frame)
    assert len([t for t in texts if t.startswith("Learning Rate ")]) == 2


def test_ctk_switching_setting_relabels_and_repairs_the_inputs(ctk_root):
    config = make_config()
    config.multi_config_mode = MultiConfigMode.SINGLE_SETTING
    config.multi_config_sweep_setting = MultiConfigSweepSetting.LEARNING_RATE
    tab = build_ctk_tab(ctk_root, config)

    tab.ui_state.get_var("multi_config_sweep_value_1").set("1e-4")
    assert config.multi_config_sweep_value_1 == "1e-4"

    tab.ui_state.get_var("multi_config_sweep_setting").set(str(MultiConfigSweepSetting.OPTIMIZER))
    tab.refresh_values_frame()

    texts = ctk_texts(tab.values_frame)
    assert len([t for t in texts if t.startswith("Optimizer ")]) == 3
    # the stale learning rate was replaced with a real optimizer name
    assert config.multi_config_sweep_value_1 in MultiConfigTabController(config).sweep_choices()


def test_ctk_adaptive_mode_shows_the_rate_input_and_preview(ctk_root):
    config = make_config()
    config.multi_config_adaptive_lr = 0.0003
    tab = build_ctk_tab(ctk_root, config)

    tab.ui_state.get_var("multi_config_mode").set(str(MultiConfigMode.ADAPTIVE_LEARNING_RATE))
    tab.refresh_mode_frame()

    texts = ctk_texts(tab.scroll_frame)
    assert "Starting Learning Rate" in texts
    assert "0.0002 / 0.0003 / 0.0004" in texts
    # the other modes' inputs are gone
    assert "Config Set 1" not in texts
    assert "Number of Values" not in texts


def test_ctk_the_preview_follows_the_typed_rate(ctk_root):
    config = make_config()
    config.multi_config_mode = MultiConfigMode.ADAPTIVE_LEARNING_RATE
    config.multi_config_adaptive_lr = 0.0003
    tab = build_ctk_tab(ctk_root, config)

    assert "0.0002 / 0.0003 / 0.0004" in ctk_texts(tab.values_frame)

    # the entry's own callback rebuilds the preview, so setting the var is the whole interaction
    tab.ui_state.get_var("multi_config_adaptive_lr").set("0.0009")

    assert "0.0008 / 0.0009 / 0.001" in ctk_texts(tab.values_frame)


def test_ctk_end_early_is_shown_in_every_mode(ctk_root):
    for mode in MultiConfigMode:
        config = make_config()
        config.multi_config_mode = mode
        tab = build_ctk_tab(ctk_root, config)
        assert "End Early" in ctk_texts(tab.scroll_frame), f"missing in {mode}"


def test_ctk_layer_filter_shows_the_independent_lineage_note(ctk_root):
    from modules.util.enum.ModelType import ModelType
    from modules.util.enum.TrainingMethod import TrainingMethod

    config = make_config()
    config.model_type = ModelType.FLUX_DEV_1
    config.training_method = TrainingMethod.LORA
    config.multi_config_mode = MultiConfigMode.SINGLE_SETTING
    config.multi_config_sweep_setting = MultiConfigSweepSetting.LAYER_FILTER
    tab = build_ctk_tab(ctk_root, config)

    texts = " ".join(ctk_texts(tab.scroll_frame))
    assert "own model" in texts
    assert "Layer Filter 1" in texts


# --- PySide6 view -------------------------------------------------------------------------------


@pytest.fixture
def qt_app():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def build_qt_tab(config):
    from modules.ui.PySide6MultiConfigTabView import PySide6MultiConfigTabView
    from modules.util.ui.PySide6UIState import PySide6UIState

    ui_state = PySide6UIState(config)
    return PySide6MultiConfigTabView(None, MultiConfigTabController(config), ui_state)


def qt_texts(widget) -> list[str]:
    from PySide6.QtWidgets import QLabel

    return [child.text() for child in widget.findChildren(QLabel) if child.text()]


def test_qt_tab_starts_in_full_config_mode(qt_app):
    config = make_config()
    tab = build_qt_tab(config)

    texts = qt_texts(tab)
    assert "Number of Configs" in texts
    for slot in range(1, config.multi_config_path_count + 1):
        assert f"Config Set {slot}" in texts
    assert f"Config Set {config.multi_config_path_count + 1}" not in texts
    assert "Selection Metric" in texts


def test_qt_tab_switches_to_the_sweep_inputs(qt_app):
    config = make_config()
    config.multi_config_sweep_setting = MultiConfigSweepSetting.LEARNING_RATE
    tab = build_qt_tab(config)

    tab.ui_state.get_var("multi_config_mode").set(str(MultiConfigMode.SINGLE_SETTING))
    tab.refresh_mode_frame()

    texts = qt_texts(tab)
    assert "Setting" in texts
    assert "Number of Values" in texts
    assert "Config Set 1" not in texts
    assert len([t for t in texts if t.startswith("Learning Rate ")]) == 3


def test_qt_value_count_changes_the_number_of_inputs(qt_app):
    config = make_config()
    config.multi_config_mode = MultiConfigMode.SINGLE_SETTING
    config.multi_config_sweep_setting = MultiConfigSweepSetting.TIMESTEP_DISTRIBUTION
    tab = build_qt_tab(config)

    tab.ui_state.get_var("multi_config_sweep_count").set("5")
    tab.refresh_values_frame()

    texts = qt_texts(tab.values_frame)
    assert len([t for t in texts if t.startswith("Timestep Distribution ")]) == 5
