"""Builds the training tab and checks the dataset curation controls are on it.

The curation frame is appended after each model type's own layout rather than being threaded through
every one of them, so nothing else would notice if it stopped appearing — or if a setting were added
to the config and never given a control. Building the real tab in both UI frameworks is what catches
that.

Run with:  pytest tests/test_training_tab_curation.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.ui.TrainingTabController import TrainingTabController
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType

import pytest

CURATION_LABELS = [
    "Detect Problem Images",
    "Per-Image Adaptive LR",
    "Warmup Epochs",
    "Plateau Patience",
    "Write Loss Log",
    "Auto-Recaption Stuck",
    "Recaption Model",
    "Recaption Precision",
    "Recaption Trigger Word",
    "Warm Up Look Outliers",
    "Outlier Start LR",
    "Outlier Ramp Epochs",
    "Outlier Sensitivity",
]


def make_config(model_type: ModelType = ModelType.STABLE_DIFFUSION_15) -> TrainConfig:
    config = TrainConfig.default_values()
    config.model_type = model_type
    return config


# --- CustomTkinter --------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ctk_root():
    ctk = pytest.importorskip("customtkinter")
    pytest.importorskip("matplotlib")  # the tab pulls in the timestep distribution window
    try:
        root = ctk.CTk()
    except Exception as e:  # no display
        pytest.skip(f"no Tk display available: {e}")
    root.withdraw()
    yield root
    root.destroy()


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


def build_ctk_tab(root, config):
    from modules.ui.CtkTrainingTabView import CtkTrainingTabView
    from modules.util.ui.CtkUIState import CtkUIState

    import customtkinter as ctk

    frame = ctk.CTkFrame(root)
    return CtkTrainingTabView(frame, TrainingTabController(config), CtkUIState(root, config))


@pytest.mark.parametrize("label", CURATION_LABELS)
def test_ctk_training_tab_offers_every_curation_setting(ctk_root, label):
    tab = build_ctk_tab(ctk_root, make_config())

    assert label in ctk_texts(tab.scroll_frame)


def test_ctk_the_curation_frame_is_there_for_every_model_type(ctk_root):
    # the frame is appended past the rows any single layout uses, so it has to survive all of them
    for model_type in (ModelType.STABLE_DIFFUSION_15, ModelType.STABLE_DIFFUSION_XL_10_BASE,
                       ModelType.FLUX_DEV_1):
        tab = build_ctk_tab(ctk_root, make_config(model_type))
        assert "Detect Problem Images" in ctk_texts(tab.scroll_frame), model_type


def test_ctk_every_captioner_is_offered(ctk_root):
    from modules.util.enum.CurationCaptionModel import CurationCaptionModel

    import customtkinter as ctk

    tab = build_ctk_tab(ctk_root, make_config())

    menus = []

    def walk(widget):
        for child in widget.winfo_children():
            if isinstance(child, ctk.CTkOptionMenu):
                menus.append(child.cget("values"))
            walk(child)

    walk(tab.scroll_frame)
    assert [str(x) for x in CurationCaptionModel] in menus


# --- PySide6 --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qt_app():
    pytest.importorskip("PySide6")
    pytest.importorskip("matplotlib")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def build_qt_tab(config):
    from modules.ui.PySide6TrainingTabView import PySide6TrainingTabView
    from modules.util.ui.PySide6UIState import PySide6UIState

    return PySide6TrainingTabView(None, TrainingTabController(config), PySide6UIState(config))


def qt_texts(widget) -> list[str]:
    from PySide6.QtWidgets import QLabel

    return [child.text() for child in widget.findChildren(QLabel) if child.text()]


@pytest.mark.parametrize("label", CURATION_LABELS)
def test_qt_training_tab_offers_every_curation_setting(qt_app, label):
    tab = build_qt_tab(make_config())

    assert label in qt_texts(tab)


def test_qt_the_curation_frame_is_there_for_every_model_type(qt_app):
    for model_type in (ModelType.STABLE_DIFFUSION_15, ModelType.STABLE_DIFFUSION_XL_10_BASE,
                       ModelType.FLUX_DEV_1):
        tab = build_qt_tab(make_config(model_type))
        assert "Detect Problem Images" in qt_texts(tab), model_type


def test_qt_every_captioner_is_offered(qt_app):
    from modules.util.enum.CurationCaptionModel import CurationCaptionModel

    from PySide6.QtWidgets import QComboBox

    tab = build_qt_tab(make_config())

    offered = [
        [box.itemText(i) for i in range(box.count())]
        for box in tab.findChildren(QComboBox)
    ]
    assert [str(x) for x in CurationCaptionModel] in offered
