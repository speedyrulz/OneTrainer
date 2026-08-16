"""Tests for the acting half of dataset curation: look scoring, recaptioning, and hand edits.

Detecting a problem image is only half the job; these are the three things curation does about one.
None of them loads a real model — CLIP and the captioners are stood in for, because what is being
checked is the arithmetic that decides who is an outlier, the bookkeeping that stops an image being
recaptioned forever, and the file handoff that lets the user overrule both while the run continues.

Run with:  pytest tests/test_curation_extras.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

torch = pytest.importorskip("torch")

from modules.util.curation import LookConsistency  # noqa: E402
from modules.util.curation.DatasetCuration import DatasetCuration  # noqa: E402
from modules.util.curation.Recaptioner import (  # noqa: E402
    MAX_ATTEMPTS,
    Recaptioner,
    invalidate_text_cache,
)
from modules.util.enum.CurationCaptionModel import CurationCaptionModel  # noqa: E402
from modules.util.TrainProgress import TrainProgress  # noqa: E402
from tests.test_dataset_curation import make_config  # noqa: E402


def make_images(tmp_path, count: int, prefix: str = "img") -> list[str]:
    paths = []
    for i in range(count):
        image = tmp_path / f"{prefix}{i:03d}.png"
        image.write_bytes(b"")
        paths.append(str(image))
    return paths


# --- look scoring: reading and writing the sidecars ----------------------------------------------


def test_scores_written_by_a_previous_run_are_reused(tmp_path):
    paths = make_images(tmp_path, 4)
    LookConsistency.save_scores(dict.fromkeys(paths, 0.9), cutoff=0.5)

    assert LookConsistency.load_cached_scores(paths) == dict.fromkeys(paths, 0.9)


def test_scores_are_kept_beside_the_images_they_describe(tmp_path):
    first = tmp_path / "concept_a"
    second = tmp_path / "concept_b"
    first.mkdir()
    second.mkdir()
    a = make_images(first, 2)
    b = make_images(second, 2)

    LookConsistency.save_scores(dict.fromkeys(a + b, 0.8), cutoff=0.5)

    # one sidecar per concept, each listing only its own images, so a concept can be moved or
    # dropped without carrying another concept's scores with it
    for directory, expected in ((first, a), (second, b)):
        path = directory / LookConsistency.SCORES_FILENAME
        assert path.exists()
        assert set(json.loads(path.read_text())["scores"]) == set(expected)


def test_a_dataset_already_scored_by_fizgig_is_not_rescored(tmp_path):
    paths = make_images(tmp_path, 3)
    # Fizgig keys its sidecar by bare filename, not by full path
    (tmp_path / LookConsistency.FIZGIG_SCORES_FILENAME).write_text(json.dumps({
        "scores": {os.path.basename(p): 0.7 for p in paths},
    }))

    assert LookConsistency.load_cached_scores(paths) == dict.fromkeys(paths, 0.7)


def test_an_unreadable_sidecar_is_ignored_rather_than_fatal(tmp_path):
    paths = make_images(tmp_path, 3)
    (tmp_path / LookConsistency.SCORES_FILENAME).write_text("{not json")

    assert LookConsistency.load_cached_scores(paths) == {}


# --- look scoring: where "unlike the rest" starts -------------------------------------------------


def test_the_cutoff_sits_below_the_middle_of_the_pack():
    scores = {f"img{i}": value for i, value in enumerate([0.80, 0.82, 0.84, 0.86, 0.88])}

    cutoff = LookConsistency.cutoff_for(scores, deviations=1.5)

    assert cutoff < 0.84  # the median
    assert cutoff == pytest.approx(0.84 - 1.5 * 1.4826 * 0.02)


def test_one_extreme_image_does_not_drag_the_cutoff_down_with_it():
    tight = {f"img{i}": value for i, value in enumerate([0.80, 0.82, 0.84, 0.86, 0.88])}
    with_outlier = dict(tight)
    with_outlier["odd"] = -0.5

    # a mean-and-standard-deviation cutoff would widen to accommodate the outlier and stop
    # catching it; a median-and-MAD one barely moves
    assert LookConsistency.cutoff_for(with_outlier) == pytest.approx(
        LookConsistency.cutoff_for(tight), abs=0.05)
    assert with_outlier["odd"] < LookConsistency.cutoff_for(with_outlier)


def test_the_odd_one_out_is_still_found_when_the_rest_score_alike():
    # a majority of identical scores leaves the median absolute deviation at exactly zero, which
    # would otherwise switch the detector off precisely when the outlier is most obvious
    scores = {f"img{i}": 0.9 for i in range(5)} | {"odd": 0.1}

    assert scores["odd"] < LookConsistency.cutoff_for(scores)


def test_a_set_that_all_looks_alike_has_no_outliers():
    identical = {f"img{i}": 0.9 for i in range(6)}

    # zero spread, so there is no "unlike" to measure
    assert LookConsistency.cutoff_for(identical) == float("-inf")


def test_too_few_images_to_say_anything():
    assert LookConsistency.cutoff_for({"a": 0.1, "b": 0.9}) == float("-inf")


# --- look scoring: the decision -------------------------------------------------------------------


def stub_scores(monkeypatch, scores: dict[str, float]):
    monkeypatch.setattr(LookConsistency, "score_images",
                        lambda paths, device: {p: scores[p] for p in paths if p in scores})


def test_the_odd_one_out_is_found_and_the_rest_are_left_alone(tmp_path, monkeypatch):
    paths = make_images(tmp_path, 6)
    stub_scores(monkeypatch, dict.fromkeys(paths, 0.9) | {paths[3]: 0.1})

    outliers, scores = LookConsistency.find_outliers(paths, torch.device("cpu"))

    assert outliers == {paths[3]}
    assert len(scores) == len(paths)


def test_a_varied_dataset_gets_no_warmup_at_all(tmp_path, monkeypatch):
    paths = make_images(tmp_path, 6)
    # no core for the odd ones to be odd against - four of six would be "outliers"
    stub_scores(monkeypatch, dict(zip(paths, [0.9, 0.88, 0.1, 0.0, -0.2, -0.4], strict=True)))

    outliers, scores = LookConsistency.find_outliers(paths, torch.device("cpu"))

    assert outliers == set()
    assert scores  # the scores are still returned, only the verdict is withheld


def test_scoring_only_ever_runs_once_for_a_dataset(tmp_path, monkeypatch):
    paths = make_images(tmp_path, 5)
    calls = []

    def counted(to_score, device):
        calls.append(list(to_score))
        return dict.fromkeys(to_score, 0.9)

    monkeypatch.setattr(LookConsistency, "score_images", counted)

    LookConsistency.find_outliers(paths, torch.device("cpu"))
    LookConsistency.find_outliers(paths, torch.device("cpu"))

    assert len(calls) == 1
    assert calls[0] == paths


def test_only_images_are_scored(tmp_path, monkeypatch):
    paths = make_images(tmp_path, 4)
    video = str(tmp_path / "clip.mp4")
    stub_scores(monkeypatch, dict.fromkeys(paths, 0.9))

    _outliers, scores = LookConsistency.find_outliers([*paths, video], torch.device("cpu"))

    assert video not in scores


def test_a_set_too_small_to_judge_is_left_alone(tmp_path, monkeypatch):
    paths = make_images(tmp_path, 2)
    monkeypatch.setattr(LookConsistency, "score_images",
                        lambda paths, device: pytest.fail("should not have scored"))

    assert LookConsistency.find_outliers(paths, torch.device("cpu")) == (set(), {})


# --- recaptioning ---------------------------------------------------------------------------------


class FakeCaptionModel:
    """Stands in for BLIP2 and friends, recording what it was asked for."""

    def __init__(self, caption: str = "a woman in a red coat"):
        self.caption = caption
        self.seen: list[str] = []

    def generate_caption(self, sample, initial_caption="", caption_prefix="", caption_postfix=""):
        self.seen.append(initial_caption)
        return self.caption


def make_recaptioner(monkeypatch, model=None, **kwargs) -> tuple[Recaptioner, FakeCaptionModel]:
    model = model or FakeCaptionModel()
    recaptioner = Recaptioner(CurationCaptionModel.BLIP2, torch.device("cpu"), **kwargs)
    monkeypatch.setattr(recaptioner, "_load_model", lambda: model)
    return recaptioner, model


def test_a_stuck_image_gets_the_caption_the_model_actually_sees(tmp_path, monkeypatch):
    [image] = make_images(tmp_path, 1)
    caption = tmp_path / "img000.txt"
    caption.write_text("a cat", encoding="utf-8")
    recaptioner, _model = make_recaptioner(monkeypatch)

    written = recaptioner.recaption([image])

    assert written == {image: "a woman in a red coat"}
    assert caption.read_text(encoding="utf-8") == "a woman in a red coat"


def test_the_caption_it_replaced_is_kept(tmp_path, monkeypatch):
    [image] = make_images(tmp_path, 1)
    (tmp_path / "img000.txt").write_text("a cat", encoding="utf-8")
    recaptioner, _model = make_recaptioner(monkeypatch)

    recaptioner.recaption([image])

    assert (tmp_path / "img000.txt.orig").read_text(encoding="utf-8") == "a cat"


def test_the_original_is_kept_from_the_first_rewrite_not_the_last(tmp_path, monkeypatch):
    [image] = make_images(tmp_path, 1)
    (tmp_path / "img000.txt").write_text("a cat", encoding="utf-8")
    recaptioner, _model = make_recaptioner(monkeypatch)

    recaptioner.recaption([image])
    recaptioner.recaption([image])

    # otherwise the second attempt would overwrite the human's caption with the first machine one
    assert (tmp_path / "img000.txt.orig").read_text(encoding="utf-8") == "a cat"


def test_the_trigger_word_goes_at_the_end(tmp_path, monkeypatch):
    [image] = make_images(tmp_path, 1)
    recaptioner, _model = make_recaptioner(monkeypatch, trigger_word="ohwx")

    written = recaptioner.recaption([image])

    assert written[image] == "a woman in a red coat, ohwx"


def test_the_second_attempt_asks_for_more_detail_than_the_first(tmp_path, monkeypatch):
    [image] = make_images(tmp_path, 1)
    recaptioner, model = make_recaptioner(monkeypatch)

    recaptioner.recaption([image])
    recaptioner.recaption([image])

    assert model.seen == ["", "a detailed description of"]


def test_an_image_is_given_exactly_two_chances(tmp_path, monkeypatch):
    [image] = make_images(tmp_path, 1)
    recaptioner, model = make_recaptioner(monkeypatch)

    for _ in range(4):
        recaptioner.recaption([image])

    assert len(model.seen) == MAX_ATTEMPTS
    assert recaptioner.spent(image)
    assert not recaptioner.attempts_left(image)


def test_an_image_that_vanished_is_skipped(tmp_path, monkeypatch):
    recaptioner, model = make_recaptioner(monkeypatch)

    assert recaptioner.recaption([str(tmp_path / "gone.png")]) == {}
    assert model.seen == []


def test_a_failed_write_does_not_burn_the_attempt(tmp_path, monkeypatch):
    [image] = make_images(tmp_path, 1)
    recaptioner, _model = make_recaptioner(monkeypatch)
    monkeypatch.setattr(recaptioner, "_write_caption", lambda key, caption: False)

    assert recaptioner.recaption([image]) == {}
    assert recaptioner.attempts_left(image)


def test_one_image_failing_does_not_stop_the_others(tmp_path, monkeypatch):
    first, second = make_images(tmp_path, 2)

    class Awkward(FakeCaptionModel):
        def generate_caption(self, sample, initial_caption="", **kwargs):
            if sample.image_filename == first:
                raise RuntimeError("out of memory")
            return self.caption

    recaptioner, _model = make_recaptioner(monkeypatch, model=Awkward())

    written = recaptioner.recaption([first, second])

    assert set(written) == {second}
    assert recaptioner.attempts_left(first)


def test_a_captioner_that_will_not_load_costs_nothing(tmp_path, monkeypatch):
    [image] = make_images(tmp_path, 1)
    recaptioner = Recaptioner(CurationCaptionModel.BLIP2, torch.device("cpu"))
    monkeypatch.setattr(recaptioner, "_load_model", lambda: (_ for _ in ()).throw(OSError("no model")))

    assert recaptioner.recaption([image]) == {}
    assert recaptioner.attempts_left(image)


# --- the text cache -------------------------------------------------------------------------------


def test_rewritten_captions_are_re_encoded_next_epoch(tmp_path):
    text_cache = tmp_path / "text"
    text_cache.mkdir()
    (text_cache / "epoch-0-0.pt").write_bytes(b"stale")
    (tmp_path / "image").mkdir()
    (tmp_path / "image" / "epoch-0-0.pt").write_bytes(b"expensive")

    assert invalidate_text_cache(str(tmp_path))

    assert not text_cache.exists()
    # the latents cost orders of magnitude more to rebuild and none of them changed
    assert (tmp_path / "image" / "epoch-0-0.pt").exists()


def test_nothing_to_clear_is_not_an_error(tmp_path):
    assert invalidate_text_cache(str(tmp_path)) is False


# --- what curation does with all of it ------------------------------------------------------------


def train_epoch(curation, keys, losses, epoch):
    progress = TrainProgress(epoch=epoch, global_step=epoch * 10)
    for step in range(10):
        progress.global_step = epoch * 10 + step
        curation.observe_step(
            progress, keys,
            torch.tensor(losses),
            {"timestep": torch.tensor([0.5] * len(keys))},
        )


def test_a_caption_edited_by_hand_gives_the_image_a_fresh_start(tmp_path):
    workspace = tmp_path / "workspace"
    images = make_images(tmp_path, 2)
    curation = DatasetCuration(make_config(cache_dir=str(tmp_path / "cache")), str(workspace))

    for epoch in range(6):
        train_epoch(curation, images, [4.0, 0.1], epoch)
        curation.epoch_boundary(TrainProgress(epoch=epoch, global_step=epoch * 10))

    stuck = curation.watch.multiplier(images[0])

    queue = workspace / "loss_log" / "caption_edits.json"
    queue.parent.mkdir(parents=True, exist_ok=True)
    queue.write_text(json.dumps([images[0]]), encoding="utf-8")

    curation.epoch_boundary(TrainProgress(epoch=6, global_step=60))

    assert curation.watch.multiplier(images[0]) == 1.0 or \
           curation.watch.multiplier(images[0]) > stuck
    assert not queue.exists()  # consumed, so the next boundary does not reset it again
    curation.close()


def test_the_edit_queue_survives_being_corrupt(tmp_path):
    workspace = tmp_path / "workspace"
    curation = DatasetCuration(make_config(), str(workspace))
    queue = workspace / "loss_log" / "caption_edits.json"
    queue.parent.mkdir(parents=True, exist_ok=True)
    queue.write_text("[half a fi", encoding="utf-8")

    assert curation._apply_caption_edits() is False
    assert not queue.exists()
    curation.close()


def test_a_hand_edit_hands_the_image_its_recaption_attempts_back(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    [image] = make_images(tmp_path, 1)
    curation = DatasetCuration(make_config(curate_recaption=True), str(workspace))
    curation.recaptioner.attempts[image] = MAX_ATTEMPTS

    queue = workspace / "loss_log" / "caption_edits.json"
    queue.parent.mkdir(parents=True, exist_ok=True)
    queue.write_text(json.dumps([image]), encoding="utf-8")

    assert curation._apply_caption_edits() is True

    # the user's caption is a new caption, so the machine gets to judge it afresh
    assert curation.recaptioner.attempts_left(image)
    curation.close()


def test_curation_recaptions_only_the_images_it_confirmed_stuck(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    images = make_images(tmp_path, 3)
    curation = DatasetCuration(make_config(curate_recaption=True), str(workspace))
    _recaptioner, model = make_recaptioner(monkeypatch)
    asked: list[list[str]] = []
    monkeypatch.setattr(curation.recaptioner, "_load_model", lambda: model)
    original = curation.recaptioner.recaption
    monkeypatch.setattr(curation.recaptioner, "recaption",
                        lambda keys: (asked.append(list(keys)), original(keys))[1])

    verdicts = {images[0]: "stuck", images[1]: "learning", images[2]: "watch"}
    assert curation._auto_recaption(verdicts) is True

    assert asked == [[images[0]]]
    curation.close()


def test_an_image_that_stays_stuck_after_both_rewrites_is_dropped(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    [image] = make_images(tmp_path, 1)
    curation = DatasetCuration(make_config(curate_recaption=True), str(workspace))
    _recaptioner, model = make_recaptioner(monkeypatch)
    monkeypatch.setattr(curation.recaptioner, "_load_model", lambda: model)

    for _ in range(MAX_ATTEMPTS):
        curation._auto_recaption({image: "stuck"})

    assert curation.recaptioner.spent(image)
    # marked incorrigible, so the next stuck verdict excludes it instead of throttling forever
    assert image in curation.watch._incorrigible
    curation.close()


def test_nothing_is_recaptioned_when_the_toggle_is_off(tmp_path):
    curation = DatasetCuration(make_config(curate_recaption=False), str(tmp_path / "workspace"))

    assert curation.recaptioner is None
    assert curation._auto_recaption({"a.png": "stuck"}) is False
    curation.close()


def test_look_outliers_ease_in_when_the_warmup_is_on(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    images = make_images(tmp_path, 6)
    stub_scores(monkeypatch, dict.fromkeys(images, 0.9) | {images[2]: 0.1})

    curation = DatasetCuration(
        make_config(curate_per_image_lr=True, curate_warmup_outliers=True,
                    curate_outlier_start=0.25, curate_outlier_ramp_epochs=4),
        str(workspace),
    )

    train_epoch(curation, images, [1.0] * 6, 0)
    curation.epoch_boundary(TrainProgress(epoch=0, global_step=10))

    assert curation.watch.multiplier(images[2]) < 1.0
    assert curation.watch.multiplier(images[0]) == pytest.approx(1.0)
    curation.close()


def test_look_scoring_failing_does_not_stop_the_run(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    images = make_images(tmp_path, 6)
    monkeypatch.setattr(
        LookConsistency, "find_outliers",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no CLIP weights")))

    curation = DatasetCuration(
        make_config(curate_per_image_lr=True, curate_warmup_outliers=True), str(workspace))

    train_epoch(curation, images, [1.0] * 6, 0)
    curation.epoch_boundary(TrainProgress(epoch=0, global_step=10))

    assert curation.watch.multiplier(images[0]) == pytest.approx(1.0)
    curation.close()


def test_no_look_scoring_happens_unless_it_is_asked_for(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    images = make_images(tmp_path, 6)
    monkeypatch.setattr(LookConsistency, "find_outliers",
                        lambda *a, **k: pytest.fail("should not have scored"))

    curation = DatasetCuration(make_config(curate_warmup_outliers=False), str(workspace))

    train_epoch(curation, images, [1.0] * 6, 0)
    curation.epoch_boundary(TrainProgress(epoch=0, global_step=10))
    curation.close()


# --- the CLIP call itself -------------------------------------------------------------------------
#
# The rest of the look tests stub score_images out entirely, which is exactly how a shape bug in the
# one function that touches the transformers API reached a user. This drives the real thing against a
# stand-in CLIP.


class StubVisionOutput(dict):
    """What transformers 5 hands back: a ModelOutput, indexable like a tuple."""

    def __init__(self, pooler_output, last_hidden_state):
        super().__init__(last_hidden_state=last_hidden_state, pooler_output=pooler_output)
        self.pooler_output = pooler_output
        self.last_hidden_state = last_hidden_state

    def __getitem__(self, key):
        if key == 0:
            return self.last_hidden_state
        return super().__getitem__(key)


class StubClip:
    def __init__(self, mode: str, vector=None):
        self.mode = mode
        self.vector = vector if vector is not None else torch.tensor([1.0, 0.0, 0.0, 0.0])

    def get_image_features(self, pixel_values=None):
        pooled = self.vector.unsqueeze(0)                       # [1, dim]
        if self.mode == "tensor":                               # transformers 4
            return pooled
        patches = torch.zeros(1, 257, self.vector.shape[0])      # [1, seq, hidden]
        return StubVisionOutput(pooled, patches)                # transformers 5


# The shape handling itself lives in tests/test_clip_util.py, next to the helper both callers
# now share. What is checked here is the scoring built on top of it.


def test_scoring_a_real_folder_end_to_end(tmp_path, monkeypatch):
    from PIL import Image as PILImage

    paths = []
    for i in range(4):
        path = tmp_path / f"img{i}.png"
        PILImage.new("RGB", (32, 32), (10 * i, 40, 60)).save(path)
        paths.append(str(path))

    # three images that look alike and one that does not
    vectors = {
        paths[0]: torch.tensor([1.0, 0.0, 0.0, 0.0]),
        paths[1]: torch.tensor([0.99, 0.1, 0.0, 0.0]),
        paths[2]: torch.tensor([0.98, 0.0, 0.1, 0.0]),
        paths[3]: torch.tensor([0.0, 0.0, 0.0, 1.0]),
    }
    order = iter(paths)
    monkeypatch.setattr(
        "transformers.CLIPModel.from_pretrained",
        classmethod(lambda cls, name, **kw: _SequencedClip(order, vectors)))

    scores = LookConsistency.score_images(paths, torch.device("cpu"))

    assert set(scores) == set(paths)
    assert all(isinstance(v, float) for v in scores.values())
    # the odd one out scores furthest from the median of the set
    assert scores[paths[3]] == min(scores.values())


class _SequencedClip(StubClip):
    """Returns the next path's vector on each call, so a whole folder can be scored."""

    def __init__(self, order, vectors):
        super().__init__("model_output")
        self.order = order
        self.vectors = vectors

    def to(self, _device):
        return self

    def eval(self):
        return self

    def get_image_features(self, pixel_values=None):
        self.vector = self.vectors[next(self.order)]
        return super().get_image_features(pixel_values=pixel_values)
