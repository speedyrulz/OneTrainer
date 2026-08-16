"""Tests for the Problem Images window: what it shows, and how it answers back.

The window and the training loop never share an object — the trainer writes its verdicts to a file
and the window leaves edited captions in another one. So the things worth testing are the two ends
of that handoff: that a report is read, ranked and explained correctly, and that saving a caption
really does reach the trainer.

Run with:  pytest tests/test_problem_images_window.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.ui.ProblemImagesWindowController import (
    VERDICT_ORDER,
    ProblemImage,
    ProblemImagesWindowController,
)
from modules.util.config.TrainConfig import TrainConfig

import pytest


def make_config(workspace) -> TrainConfig:
    config = TrainConfig.default_values()
    config.workspace_dir = str(workspace)
    return config


def make_image(directory, name: str, caption: str = "a photo") -> str:
    from PIL import Image

    path = directory / f"{name}.png"
    Image.new("RGB", (8, 8), (120, 40, 40)).save(path)
    (directory / f"{name}.txt").write_text(caption, encoding="utf-8")
    return str(path)


def write_report(workspace, images: dict, **top_level) -> str:
    path = workspace / "loss_log" / "problem_images.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"epoch": 7, "apply_lr": True, "images": images}
    payload.update(top_level)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


@pytest.fixture
def dataset(tmp_path):
    """A small workspace with three images and a report about them."""
    images = tmp_path / "images"
    workspace = tmp_path / "workspace"
    images.mkdir()

    keys = {
        "stuck": make_image(images, "stuck", "a cat"),
        "learning": make_image(images, "learning", "a dog"),
        "fine": make_image(images, "fine", "a bird"),
    }
    write_report(workspace, {
        keys["learning"]: {"verdict": "learning", "multiplier": 1.0, "mean_residual": 0.4,
                           "slope": -0.05, "improving": True, "epochs": 6},
        keys["stuck"]: {"verdict": "stuck", "multiplier": 0.5, "mean_residual": 1.2,
                        "slope": 0.01, "improving": False, "epochs": 6, "stuck_epochs": 3},
        keys["fine"]: {"verdict": "healthy", "multiplier": 1.0, "mean_residual": -0.3},
    })
    return workspace, keys


# --- reading the report ---------------------------------------------------------------------------


def test_nothing_to_read_yet(tmp_path):
    controller = ProblemImagesWindowController(make_config(tmp_path))

    assert controller.load() is False
    assert "Detect Problem Images" in controller.status_text()


def test_a_report_with_no_problems_says_so(tmp_path):
    workspace = tmp_path / "workspace"
    write_report(workspace, {})
    controller = ProblemImagesWindowController(make_config(workspace))

    assert controller.load() is False
    assert "training normally" in controller.status_text()


def test_only_the_images_worth_looking_at_are_shown(dataset):
    workspace, keys = dataset
    controller = ProblemImagesWindowController(make_config(workspace))

    controller.load()

    shown = {image.key for image in controller.images}
    assert shown == {keys["stuck"], keys["learning"]}


def test_the_worst_image_is_at_the_top(dataset):
    workspace, keys = dataset
    controller = ProblemImagesWindowController(make_config(workspace))

    controller.load()

    assert [image.key for image in controller.images] == [keys["stuck"], keys["learning"]]


def test_images_with_the_same_verdict_are_ranked_by_how_hard_they_are(tmp_path):
    workspace = tmp_path / "workspace"
    images = tmp_path / "images"
    images.mkdir()
    mild = make_image(images, "mild")
    severe = make_image(images, "severe")
    write_report(workspace, {
        mild: {"verdict": "stuck", "mean_residual": 0.4},
        severe: {"verdict": "stuck", "mean_residual": 2.0},
    })
    controller = ProblemImagesWindowController(make_config(workspace))

    controller.load()

    assert [image.key for image in controller.images] == [severe, mild]


def test_every_verdict_the_trainer_emits_has_an_explanation():
    from modules.ui.ProblemImagesWindowController import VERDICT_HELP

    assert set(VERDICT_HELP) == set(VERDICT_ORDER)


def test_a_half_written_report_is_not_shown(tmp_path):
    workspace = tmp_path / "workspace"
    path = workspace / "loss_log" / "problem_images.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"images": {"a.png": {"verd', encoding="utf-8")
    controller = ProblemImagesWindowController(make_config(workspace))

    assert controller.load() is False
    assert controller.images == []


def test_the_status_line_counts_each_verdict(dataset):
    workspace, _keys = dataset
    controller = ProblemImagesWindowController(make_config(workspace))

    controller.load()

    assert controller.status_text() == "1 stuck, 1 learning after epoch 7"


def test_the_window_notices_a_new_report(dataset):
    workspace, keys = dataset
    controller = ProblemImagesWindowController(make_config(workspace))

    controller.load()
    assert controller.has_changed() is False

    os.utime(controller.report_path, (1_600_000_000, 1_600_000_000))
    assert controller.has_changed() is True


def test_a_report_appearing_mid_run_is_noticed(tmp_path):
    workspace = tmp_path / "workspace"
    controller = ProblemImagesWindowController(make_config(workspace))
    controller.load()

    write_report(workspace, {})

    assert controller.has_changed() is True


# --- the plateau banner --------------------------------------------------------------------------


def test_a_run_that_is_still_learning_shows_no_banner(dataset):
    workspace, _keys = dataset
    controller = ProblemImagesWindowController(make_config(workspace))

    controller.load()

    assert controller.plateau_text() is None


def test_a_confirmed_plateau_names_the_epoch_to_look_at(tmp_path):
    workspace = tmp_path / "workspace"
    write_report(workspace, {}, plateaued=True, pending_count=0, best_epoch_estimate=42)
    controller = ProblemImagesWindowController(make_config(workspace))

    controller.load()
    text = controller.plateau_text()

    assert "confirmed" in text
    assert "epoch 42" in text


def test_a_plateau_with_images_still_being_judged_is_flagged_as_provisional(tmp_path):
    workspace = tmp_path / "workspace"
    write_report(workspace, {}, plateaued=True, pending_count=3, best_epoch_estimate=42)
    controller = ProblemImagesWindowController(make_config(workspace))

    controller.load()

    assert "provisional" in controller.plateau_text()


# --- one row --------------------------------------------------------------------------------------


def test_a_row_finds_its_caption_beside_the_image(dataset):
    _workspace, keys = dataset
    image = ProblemImage(keys["stuck"], {"verdict": "stuck"})

    assert image.exists
    assert image.read_caption() == "a cat"


def test_a_row_for_an_image_that_has_since_been_deleted(tmp_path):
    image = ProblemImage(str(tmp_path / "gone.png"), {"verdict": "stuck"})

    assert not image.exists
    assert image.read_caption() == ""


def test_the_summary_says_what_was_decided_and_what_was_done(dataset):
    _workspace, keys = dataset
    image = ProblemImage(keys["stuck"], {
        "verdict": "stuck", "multiplier": 0.5, "mean_residual": 1.2,
        "slope": 0.01, "improving": False, "epochs": 6, "stuck_epochs": 3,
    })

    summary = image.summary()

    assert summary.startswith("stuck")
    assert "x0.50" in summary
    assert "flat or worsening" in summary
    assert "stuck 3 epochs" in summary


def test_an_untouched_image_is_not_described_as_throttled(dataset):
    _workspace, keys = dataset
    image = ProblemImage(keys["learning"], {"verdict": "learning", "multiplier": 1.0,
                                            "slope": -0.05, "improving": True})

    summary = image.summary()

    assert "training at" not in summary
    assert "improving" in summary


# --- writing back ----------------------------------------------------------------------------------


def test_an_edited_caption_reaches_both_the_image_and_the_trainer(dataset):
    workspace, keys = dataset
    controller = ProblemImagesWindowController(make_config(workspace))
    controller.load()
    image = controller.images[0]

    assert controller.save_caption(image, "  a tabby cat on a windowsill  ") is None

    assert image.read_caption() == "a tabby cat on a windowsill"
    queued = json.loads((workspace / "loss_log" / "caption_edits.json").read_text())
    assert queued == [keys["stuck"]]


def test_an_empty_caption_is_refused(dataset):
    workspace, _keys = dataset
    controller = ProblemImagesWindowController(make_config(workspace))
    controller.load()
    image = controller.images[0]
    before = image.read_caption()

    assert controller.save_caption(image, "   ") == "The caption is empty."
    assert image.read_caption() == before


def test_editing_two_images_before_the_next_boundary_queues_both(dataset):
    workspace, keys = dataset
    controller = ProblemImagesWindowController(make_config(workspace))
    controller.load()

    controller.save_caption(controller.images[0], "one")
    controller.save_caption(controller.images[1], "two")

    queued = json.loads((workspace / "loss_log" / "caption_edits.json").read_text())
    assert set(queued) == {keys["stuck"], keys["learning"]}


def test_editing_the_same_image_twice_queues_it_once(dataset):
    workspace, _keys = dataset
    controller = ProblemImagesWindowController(make_config(workspace))
    controller.load()

    controller.save_caption(controller.images[0], "one")
    controller.save_caption(controller.images[0], "two after thinking about it")

    queued = json.loads((workspace / "loss_log" / "caption_edits.json").read_text())
    assert len(queued) == 1


def test_a_corrupt_queue_does_not_lose_the_edit_being_made(dataset):
    workspace, keys = dataset
    controller = ProblemImagesWindowController(make_config(workspace))
    controller.load()
    queue = workspace / "loss_log" / "caption_edits.json"
    queue.parent.mkdir(parents=True, exist_ok=True)
    queue.write_text("[half a fi", encoding="utf-8")

    assert controller.save_caption(controller.images[0], "a tabby cat") is None

    assert json.loads(queue.read_text()) == [keys["stuck"]]


def test_a_caption_that_cannot_be_written_says_why(dataset, monkeypatch):
    workspace, _keys = dataset
    controller = ProblemImagesWindowController(make_config(workspace))
    controller.load()

    def refuse(*args, **kwargs):
        raise PermissionError("read-only")

    monkeypatch.setattr("builtins.open", refuse)

    error = controller.save_caption(controller.images[0], "a tabby cat")

    assert error is not None
    assert "stuck.txt" in error


# --- the window, in both UI frameworks -------------------------------------------------------------


@pytest.fixture(scope="module")
def ctk_root():
    # one root for the whole module: each window here is a Toplevel, and repeatedly building and
    # tearing down Tk roots in a single process is what makes these suites flaky
    ctk = pytest.importorskip("customtkinter")
    try:
        root = ctk.CTk()
    except Exception as e:  # no display
        pytest.skip(f"no Tk display available: {e}")
    root.withdraw()
    yield root
    root.destroy()


def build_ctk_window(root, workspace):
    from modules.ui.CtkProblemImagesWindowView import CtkProblemImagesWindowView

    controller = ProblemImagesWindowController(make_config(workspace))
    return CtkProblemImagesWindowView(root, controller)


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


def test_ctk_window_lists_one_row_per_flagged_image(ctk_root, dataset):
    workspace, _keys = dataset
    window = build_ctk_window(ctk_root, workspace)

    assert len(window._editors) == 2
    texts = ctk_texts(window.scroll)
    assert "stuck.png" in texts
    assert "learning.png" in texts
    assert "fine.png" not in texts

    window.destroy()


def test_ctk_window_shows_each_caption_ready_to_edit(ctk_root, dataset):
    workspace, keys = dataset
    window = build_ctk_window(ctk_root, workspace)

    assert window._editors[keys["stuck"]].get("1.0", "end").strip() == "a cat"

    window.destroy()


def test_ctk_window_explains_what_a_verdict_means(ctk_root, dataset):
    workspace, _keys = dataset
    window = build_ctk_window(ctk_root, workspace)

    texts = " ".join(ctk_texts(window.scroll))
    assert "caption does not match" in texts

    window.destroy()


def test_ctk_editing_a_caption_saves_it(ctk_root, dataset):
    workspace, keys = dataset
    window = build_ctk_window(ctk_root, workspace)

    editor = window._editors[keys["stuck"]]
    editor.delete("1.0", "end")
    editor.insert("1.0", "a tabby cat on a windowsill")
    window.controller.save_caption(window.controller.images[0], editor.get("1.0", "end"))

    assert (workspace.parent / "images" / "stuck.txt").read_text(encoding="utf-8") \
        == "a tabby cat on a windowsill"

    window.destroy()


def test_ctk_a_caption_being_typed_survives_a_refresh(ctk_root, dataset):
    workspace, keys = dataset
    window = build_ctk_window(ctk_root, workspace)

    editor = window._editors[keys["stuck"]]
    editor.delete("1.0", "end")
    editor.insert("1.0", "half a thou")

    window.refresh()  # the trainer wrote a new report mid-sentence

    assert window._editors[keys["stuck"]].get("1.0", "end").strip() == "half a thou"

    window.destroy()


def test_ctk_window_with_nothing_to_show(ctk_root, tmp_path):
    window = build_ctk_window(ctk_root, tmp_path / "workspace")

    assert window._editors == {}
    assert "Detect Problem Images" in window.status_label.cget("text")

    window.destroy()


def test_ctk_window_shows_the_plateau_banner(ctk_root, tmp_path):
    workspace = tmp_path / "workspace"
    write_report(workspace, {}, plateaued=True, pending_count=0, best_epoch_estimate=42)
    window = build_ctk_window(ctk_root, workspace)

    assert "epoch 42" in window.plateau_label.cget("text")

    window.destroy()


@pytest.fixture
def qt_app():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def build_qt_window(workspace):
    from modules.ui.PySide6ProblemImagesWindowView import PySide6ProblemImagesWindowView

    controller = ProblemImagesWindowController(make_config(workspace))
    return PySide6ProblemImagesWindowView(None, controller)


def qt_texts(widget) -> list[str]:
    from PySide6.QtWidgets import QLabel

    return [child.text() for child in widget.findChildren(QLabel) if child.text()]


def test_qt_window_lists_one_row_per_flagged_image(qt_app, dataset):
    workspace, _keys = dataset
    window = build_qt_window(workspace)

    assert len(window._editors) == 2
    texts = qt_texts(window)
    assert "stuck.png" in texts
    assert "learning.png" in texts
    assert "fine.png" not in texts

    window.close()


def test_qt_window_shows_each_caption_ready_to_edit(qt_app, dataset):
    workspace, keys = dataset
    window = build_qt_window(workspace)

    assert window._editors[keys["stuck"]].toPlainText() == "a cat"

    window.close()


def test_qt_editing_a_caption_saves_it(qt_app, dataset):
    workspace, _keys = dataset
    window = build_qt_window(workspace)

    window.controller.save_caption(window.controller.images[0], "a tabby cat on a windowsill")

    assert (workspace.parent / "images" / "stuck.txt").read_text(encoding="utf-8") \
        == "a tabby cat on a windowsill"

    window.close()


def test_qt_a_caption_being_typed_survives_a_refresh(qt_app, dataset):
    workspace, keys = dataset
    window = build_qt_window(workspace)

    window._editors[keys["stuck"]].setPlainText("half a thou")
    window.refresh()

    assert window._editors[keys["stuck"]].toPlainText() == "half a thou"

    window.close()


def test_qt_rows_from_the_previous_report_are_gone_after_a_refresh(qt_app, dataset):
    workspace, keys = dataset
    window = build_qt_window(workspace)

    write_report(workspace, {
        keys["stuck"]: {"verdict": "stuck", "mean_residual": 1.2},
    })
    window.refresh()

    assert set(window._editors) == {keys["stuck"]}
    assert "learning.png" not in qt_texts(window)

    window.close()


def test_qt_window_with_nothing_to_show(qt_app, tmp_path):
    window = build_qt_window(tmp_path / "workspace")

    assert window._editors == {}
    assert "Detect Problem Images" in window.status_label.text()

    window.close()


def test_qt_window_shows_the_plateau_banner(qt_app, tmp_path):
    workspace = tmp_path / "workspace"
    write_report(workspace, {}, plateaued=True, pending_count=0, best_epoch_estimate=42)
    window = build_qt_window(workspace)

    assert "epoch 42" in window.plateau_label.text()

    window.close()
