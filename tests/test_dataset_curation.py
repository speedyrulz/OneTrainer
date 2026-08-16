"""Tests for dataset curation: the per-image loss watch and the trainer bridge.

The watch decides which images are hurting the run; the bridge turns a training step into something
it can read and its verdicts into a weight per sample. What matters is that the two agree about
which image is which, that the weighting really is a per-image learning rate, and that an excluded
image contributes nothing.

Run with:  pytest tests/test_dataset_curation.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

torch = pytest.importorskip("torch")

from modules.util.config.TrainConfig import TrainConfig  # noqa: E402
from modules.util.curation.DatasetCuration import DatasetCuration, curation_enabled  # noqa: E402
from modules.util.curation.loss_watch import PerImageLossWatch  # noqa: E402
from modules.util.TrainProgress import TrainProgress  # noqa: E402


def make_config(**overrides) -> TrainConfig:
    config = TrainConfig.default_values()
    config.curate_dataset = True
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def make_curation(tmp_path, **overrides) -> DatasetCuration:
    return DatasetCuration(make_config(**overrides), str(tmp_path))


def make_images(tmp_path, count: int, caption: str = "a photo") -> list[str]:
    paths = []
    for i in range(count):
        image = tmp_path / f"img{i:03d}.png"
        image.write_bytes(b"")
        (tmp_path / f"img{i:03d}.txt").write_text(caption, encoding="utf-8")
        paths.append(str(image))
    return paths


# --- switching it on --------------------------------------------------------------------------


def test_curation_is_off_by_default():
    assert not curation_enabled(TrainConfig.default_values())


@pytest.mark.parametrize("field", ["curate_dataset", "curate_per_image_lr"])
def test_either_toggle_turns_it_on(field):
    config = TrainConfig.default_values()
    setattr(config, field, True)
    assert curation_enabled(config)


# --- reading a training step ------------------------------------------------------------------


def test_image_keys_come_straight_from_the_batch():
    batch = {"image_path": ["a/1.png", "a/2.png"]}
    assert DatasetCuration.image_keys(batch) == ["a/1.png", "a/2.png"]


def test_a_batch_without_image_paths_is_ignored():
    assert DatasetCuration.image_keys({}) is None


def test_flow_matching_timesteps_pass_through():
    data = {"timestep": torch.tensor([0.1, 0.9])}
    assert DatasetCuration.normalised_timesteps(data, 2) == pytest.approx([0.1, 0.9])


def test_integer_schedule_timesteps_are_scaled_into_range():
    # a diffusion model hands over indices into its schedule, not sigmas
    data = {"timestep": torch.tensor([250.0, 750.0]), "num_train_timesteps": 1000}
    assert DatasetCuration.normalised_timesteps(data, 2) == pytest.approx([0.25, 0.75])


def test_timesteps_are_clamped_into_the_unit_range():
    data = {"timestep": torch.tensor([-0.2, 1.4])}
    assert DatasetCuration.normalised_timesteps(data, 2) == pytest.approx([0.0, 1.0])


def test_a_shared_timestep_is_spread_across_the_batch():
    data = {"timestep": torch.tensor([0.5])}
    assert DatasetCuration.normalised_timesteps(data, 3) == pytest.approx([0.5, 0.5, 0.5])


def test_a_step_without_timesteps_records_nothing(tmp_path):
    curation = make_curation(tmp_path)
    curation.observe_step(TrainProgress(), ["a.png"], torch.tensor([0.5]), {})
    assert curation.watch._records == []


# --- the weighting is a per-image learning rate -------------------------------------------------


def test_every_image_starts_unweighted(tmp_path):
    curation = make_curation(tmp_path)
    weights = curation.sample_weights(["a.png", "b.png"], torch.device("cpu"), torch.float32)
    assert torch.equal(weights, torch.ones(2))


def test_a_throttled_image_gets_a_smaller_share_of_the_step(tmp_path):
    curation = make_curation(tmp_path, curate_per_image_lr=True)
    curation.watch._mult["bad.png"] = 0.5

    weights = curation.sample_weights(["bad.png", "good.png"], torch.device("cpu"), torch.float32)
    assert weights.tolist() == [0.5, 1.0]


def test_verdicts_do_not_move_the_weights_when_per_image_lr_is_off(tmp_path):
    # detection alone is observational: it reports, it does not act
    curation = make_curation(tmp_path, curate_per_image_lr=False)
    curation.watch._mult["bad.png"] = 0.5

    weights = curation.sample_weights(["bad.png"], torch.device("cpu"), torch.float32)
    assert weights.tolist() == [1.0]


def test_an_excluded_image_contributes_no_gradient(tmp_path):
    curation = make_curation(tmp_path, curate_per_image_lr=True)
    curation.watch._excluded.add("dead.png")

    weights = curation.sample_weights(["dead.png", "live.png"], torch.device("cpu"), torch.float32)
    assert weights.tolist() == [0.0, 1.0]

    # zero weight really does mean no gradient reaches the parameter through that sample
    param = torch.nn.Parameter(torch.ones(2))
    sample_losses = param * torch.tensor([3.0, 5.0])
    (sample_losses * weights).mean().backward()
    assert param.grad[0].item() == 0.0
    assert param.grad[1].item() != 0.0


def test_weighting_scales_the_step_rather_than_renormalising_it(tmp_path):
    """Halving one image's weight must halve its pull, not redistribute it to the others."""
    curation = make_curation(tmp_path, curate_per_image_lr=True)
    curation.watch._mult["a.png"] = 0.5

    weights = curation.sample_weights(["a.png", "b.png"], torch.device("cpu"), torch.float32)
    sample_losses = torch.tensor([2.0, 2.0])

    assert (sample_losses * weights).mean().item() == pytest.approx(1.5)
    assert sample_losses.mean().item() == pytest.approx(2.0)


def test_an_excluded_image_is_not_even_measured(tmp_path):
    curation = make_curation(tmp_path)
    curation.watch._excluded.add("dead.png")

    curation.observe_step(
        TrainProgress(), ["dead.png", "live.png"],
        torch.tensor([9.0, 1.0]), {"timestep": torch.tensor([0.5, 0.5])},
    )

    recorded = {key for key, _epoch, _bucket, _loss in curation.watch._records}
    assert recorded == {"live.png"}, "an excluded image must not pad the loss statistics"


# --- observing across a batch -------------------------------------------------------------------


def test_each_image_in_a_batch_is_recorded_separately(tmp_path):
    curation = make_curation(tmp_path)
    keys = ["a.png", "b.png", "c.png"]

    curation.observe_step(
        TrainProgress(), keys,
        torch.tensor([0.1, 0.2, 0.3]),
        {"timestep": torch.tensor([0.2, 0.4, 0.6])},
    )

    recorded = {key: loss for key, _epoch, _bucket, loss in curation.watch._records}
    assert recorded == pytest.approx({"a.png": 0.1, "b.png": 0.2, "c.png": 0.3})
    # a batched run must not collapse into one composite key, which is what disables per-image LR
    assert curation.watch._batched is False


def test_the_recorded_loss_is_the_unweighted_one(tmp_path):
    """Feeding weighted losses back would let a throttled image look like it had improved."""
    curation = make_curation(tmp_path, curate_per_image_lr=True)
    curation.watch._mult["a.png"] = 0.1

    curation.observe_step(
        TrainProgress(), ["a.png"], torch.tensor([2.0]), {"timestep": torch.tensor([0.5])},
    )

    assert curation.watch._records[0][3] == pytest.approx(2.0)


def test_the_epoch_recorded_is_the_one_being_trained(tmp_path):
    curation = make_curation(tmp_path)
    progress = TrainProgress(epoch=3, global_step=120)

    curation.observe_step(progress, ["a.png"], torch.tensor([0.5]), {"timestep": torch.tensor([0.5])})

    # the watch counts epochs from 1; TrainProgress counts from 0
    assert curation.watch._records[0][1] == 4


# --- verdicts ------------------------------------------------------------------------------------


def feed(curation: DatasetCuration, keys, losses_by_epoch, timestep: float = 0.5):
    """Run several epochs where each key gets a fixed loss per epoch."""
    step = 0
    for epoch_index, losses in enumerate(losses_by_epoch):
        progress = TrainProgress(epoch=epoch_index, global_step=step)
        for key, loss in zip(keys, losses, strict=True):
            curation.observe_step(
                progress, [key], torch.tensor([loss]), {"timestep": torch.tensor([timestep])},
            )
            step += 1
        curation.epoch_boundary(TrainProgress(epoch=epoch_index + 1, global_step=step))


def test_an_image_that_stays_hard_is_flagged_and_one_that_learns_is_not(tmp_path):
    curation = make_curation(tmp_path, curate_per_image_lr=True, curate_warmup_epochs=1)
    keys = ["stuck.png", "learner.png", "easy.png"]

    # stuck stays high and flat, learner starts high and descends, easy is always low
    epochs = [[0.90, 0.90, 0.10], [0.90, 0.80, 0.10], [0.91, 0.70, 0.10],
              [0.90, 0.55, 0.10], [0.90, 0.40, 0.10], [0.91, 0.30, 0.10],
              [0.90, 0.22, 0.10], [0.90, 0.16, 0.10], [0.91, 0.12, 0.10]]
    feed(curation, keys, epochs)

    verdicts = curation.watch.verdicts
    assert verdicts.get("stuck.png") == "stuck"
    assert verdicts.get("learner.png") != "stuck", "a hard image that is descending is not stuck"
    assert curation.watch.multiplier("stuck.png") < 1.0
    assert curation.watch.multiplier("learner.png") == pytest.approx(1.0)


def test_verdicts_reach_the_problem_images_file(tmp_path):
    curation = make_curation(tmp_path, curate_warmup_epochs=1)
    keys = ["stuck.png", "fine.png"]
    feed(curation, keys, [[0.9, 0.1]] * 8)

    report = tmp_path / "loss_log" / "problem_images.json"
    assert report.exists(), "the epoch boundary should write a report"

    data = json.loads(report.read_text(encoding="utf-8"))
    assert data, "the report should not be empty once verdicts exist"


def test_nothing_is_judged_during_the_warmup(tmp_path):
    curation = make_curation(tmp_path, curate_warmup_epochs=5)
    feed(curation, ["a.png", "b.png"], [[0.9, 0.1]] * 3)

    assert curation.watch.verdicts == {}
    assert curation.watch.multiplier("a.png") == pytest.approx(1.0)


# --- exclusions travel with the images ------------------------------------------------------------


def test_an_exclusion_is_written_beside_the_image(tmp_path):
    images = make_images(tmp_path, 3)
    watch = PerImageLossWatch(str(tmp_path))
    watch._record_exclusion(images[0], epoch=4)

    sidecar = tmp_path / "excluded_images.json"
    assert sidecar.exists()
    assert images[0] in json.loads(sidecar.read_text(encoding="utf-8"))


def test_a_later_run_honours_the_exclusion(tmp_path):
    images = make_images(tmp_path, 3)
    PerImageLossWatch(str(tmp_path))._record_exclusion(images[0], epoch=4)

    resumed = PerImageLossWatch(str(tmp_path))
    resumed.preflight(images)

    assert resumed.is_excluded(images[0])
    assert not resumed.is_excluded(images[1])


def test_fixing_the_caption_re_admits_the_image(tmp_path):
    images = make_images(tmp_path, 3)
    PerImageLossWatch(str(tmp_path))._record_exclusion(images[0], epoch=4)

    caption = tmp_path / (os.path.splitext(os.path.basename(images[0]))[0] + ".txt")
    caption.write_text("a photo, now describing what is actually there", encoding="utf-8")

    resumed = PerImageLossWatch(str(tmp_path))
    resumed.preflight(images)

    assert not resumed.is_excluded(images[0]), "a changed caption should pardon the image"


def test_exclusions_are_kept_per_concept_directory(tmp_path):
    # a OneTrainer run can train several concepts at once; each keeps its own sidecar
    first, second = tmp_path / "faces", tmp_path / "cars"
    first.mkdir()
    second.mkdir()
    face_images = make_images(first, 2)
    car_images = make_images(second, 2)

    watch = PerImageLossWatch(str(tmp_path))
    watch._record_exclusion(face_images[0], epoch=3)
    watch._record_exclusion(car_images[0], epoch=3)

    assert (first / "excluded_images.json").exists()
    assert (second / "excluded_images.json").exists()
    assert list(json.loads((first / "excluded_images.json").read_text())) == [face_images[0]]

    resumed = PerImageLossWatch(str(tmp_path))
    resumed.preflight(face_images + car_images)
    assert resumed.is_excluded(face_images[0])
    assert resumed.is_excluded(car_images[0])


def test_a_run_is_never_left_with_nothing_to_train_on(tmp_path):
    images = make_images(tmp_path, 2)
    watch = PerImageLossWatch(str(tmp_path))
    for image in images:
        watch._record_exclusion(image, epoch=3)

    resumed = PerImageLossWatch(str(tmp_path))
    resumed.preflight(images)

    assert not all(resumed.is_excluded(image) for image in images), \
        "excluding every image would train on nothing"


def test_stale_entries_for_removed_images_are_pruned(tmp_path):
    images = make_images(tmp_path, 3)
    watch = PerImageLossWatch(str(tmp_path))
    watch._record_exclusion(images[0], epoch=3)

    # the image is gone from the dataset on the next run
    resumed = PerImageLossWatch(str(tmp_path))
    resumed.preflight(images[1:])

    assert not resumed._excl_data, "an entry for an image no longer trained should not linger"


# --- warmup ramp ----------------------------------------------------------------------------------


def test_warmed_up_images_ease_in(tmp_path):
    curation = make_curation(tmp_path, curate_per_image_lr=True)
    curation.set_warmup_keys(["odd.png"], start=0.4, ramp_epochs=4)
    curation.watch._epoch_now = 1

    assert curation.watch.multiplier("odd.png") < 1.0
    assert curation.watch.multiplier("normal.png") == pytest.approx(1.0)

    curation.watch._epoch_now = 10
    assert curation.watch.multiplier("odd.png") == pytest.approx(1.0), "the ramp should finish"
