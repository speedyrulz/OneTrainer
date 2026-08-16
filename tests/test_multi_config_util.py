"""Tests for the multi-config validation tournament's decision logic.

Run with:  pytest tests/test_multi_config_util.py

These cover the parts that decide what a candidate trains with, how long it trains for, and which
candidate wins - everything that can be checked without loading a model.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.util.config.TrainConfig import MULTI_CONFIG_SLOT_COUNT, MULTI_CONFIG_SWEEP_MAX, TrainConfig
from modules.util.enum.EMAMode import EMAMode
from modules.util.enum.LearningRateScheduler import LearningRateScheduler
from modules.util.enum.ModelType import ModelType
from modules.util.enum.MultiConfigMode import MultiConfigMode
from modules.util.enum.MultiConfigSelection import MultiConfigSelection
from modules.util.enum.MultiConfigSweepSetting import MultiConfigSweepSetting
from modules.util.enum.Optimizer import Optimizer
from modules.util.enum.TimestepDistribution import TimestepDistribution
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.multi_config_ladder import LearningRateLadder
from modules.util.multi_config_sweep import (
    SWEEP_SETTINGS,
    SweepValueKind,
    default_sweep_values,
    sweep_options,
    validate_sweep_value,
)
from modules.util.multi_config_util import (
    SegmentSchedule,
    build_ladder_candidates,
    load_candidate_configs,
    merge_candidate_dict,
    pick_winner,
    preflight_errors,
    score_validation_result,
    validate_before_start,
)
from modules.util.TrainProgress import TrainProgress
from modules.util.ValidationResult import ValidationResult

import pytest

# --- helpers -----------------------------------------------------------------------------------


def make_base_config(**overrides) -> TrainConfig:
    config = TrainConfig.default_values()
    config.validation = True
    config.validate_after = 1
    config.validate_after_unit = TimeUnit.EPOCH
    config.multi_config = True
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def write_config(tmp_path, name: str, **overrides) -> str:
    config = TrainConfig.default_values()
    for field, value in overrides.items():
        if "." in field:
            part, sub = field.split(".", 1)
            setattr(getattr(config, part), sub, value)
        else:
            setattr(config, field, value)
    path = os.path.join(str(tmp_path), f"{name}.json")
    with open(path, "w") as f:
        json.dump(config.to_settings_dict(secrets=False), f)
    return path


def result(per_concept: dict[str, float], counts: dict[str, int] | None = None) -> ValidationResult:
    counts = counts or dict.fromkeys(per_concept, 1)
    total = sum(per_concept[label] * counts[label] for label in per_concept)
    return ValidationResult(
        per_concept=dict(per_concept),
        sample_counts=dict(counts),
        total_average=total / sum(counts.values()),
    )


# --- SegmentSchedule ---------------------------------------------------------------------------


def test_step_segment_ends_after_the_configured_number_of_steps():
    schedule = SegmentSchedule(10, TimeUnit.STEP)
    progress = TrainProgress(epoch=0, global_step=40)
    schedule.begin(progress, now=0.0)

    for _ in range(9):
        progress.next_step(1)
        assert not schedule.is_complete(progress, now=0.0)

    progress.next_step(1)
    assert schedule.is_complete(progress, now=0.0)
    assert schedule.steps_taken(progress) == 10


def test_step_segment_does_not_end_immediately_at_a_multiple_of_the_interval():
    # the built-in validation trigger fires on global_step % interval == 0, which at the start of a
    # segment means "after one step". The tournament has to measure forward instead, or the first
    # round would compare candidates after a single step.
    schedule = SegmentSchedule(10, TimeUnit.STEP)
    progress = TrainProgress(epoch=0, global_step=100)  # already a multiple of 10
    schedule.begin(progress, now=0.0)

    progress.next_step(1)
    assert not schedule.is_complete(progress, now=0.0)


def test_epoch_segment_ends_on_the_epoch_boundary():
    schedule = SegmentSchedule(1, TimeUnit.EPOCH)
    progress = TrainProgress(epoch=3, global_step=300)
    schedule.begin(progress, now=0.0)

    for _ in range(50):
        progress.next_step(1)
        assert not schedule.is_complete(progress, now=0.0)

    progress.next_epoch()
    assert schedule.is_complete(progress, now=0.0)


def test_multi_epoch_segment_spans_the_configured_number_of_epochs():
    schedule = SegmentSchedule(3, TimeUnit.EPOCH)
    progress = TrainProgress(epoch=0)
    schedule.begin(progress, now=0.0)

    progress.next_epoch()
    assert not schedule.is_complete(progress, now=0.0)
    progress.next_epoch()
    assert not schedule.is_complete(progress, now=0.0)
    progress.next_epoch()
    assert schedule.is_complete(progress, now=0.0)


def test_wall_clock_segment_needs_both_the_time_and_at_least_one_step():
    schedule = SegmentSchedule(1, TimeUnit.MINUTE)
    progress = TrainProgress(epoch=0, global_step=0)
    schedule.begin(progress, now=1000.0)

    # time is up but nothing was trained yet
    assert not schedule.is_complete(progress, now=1061.0)

    progress.next_step(1)
    assert not schedule.is_complete(progress, now=1030.0)
    assert schedule.is_complete(progress, now=1061.0)


def test_locking_makes_every_later_candidate_train_the_same_number_of_steps():
    # This is what keeps a wall-clock interval fair: the first candidate defines the round's length in
    # steps, and a slower candidate is then judged after the same amount of training, not less.
    schedule = SegmentSchedule(1, TimeUnit.MINUTE)

    first = TrainProgress(epoch=0, global_step=0)
    schedule.begin(first, now=0.0)
    for _ in range(37):
        first.next_step(1)
    assert schedule.is_complete(first, now=61.0)
    schedule.lock_to(first)

    second = TrainProgress(epoch=0, global_step=0)
    schedule.begin(second, now=0.0)
    for _ in range(36):
        second.next_step(1)
        # a slower candidate must not be cut short just because the clock ran out
        assert not schedule.is_complete(second, now=10_000.0)
    second.next_step(1)
    assert schedule.is_complete(second, now=0.0)
    assert schedule.steps_taken(second) == 37


def test_unlock_restores_the_configured_interval():
    schedule = SegmentSchedule(5, TimeUnit.MINUTE)
    progress = TrainProgress()
    schedule.begin(progress, now=0.0)
    for _ in range(2):
        progress.next_step(1)
    assert schedule.lock_to(progress) is True
    assert "2 step" in schedule.describe()

    schedule.unlock()
    assert schedule.locked_target is None
    assert "5 minute" in schedule.describe().lower()


@pytest.mark.parametrize("unit", [TimeUnit.STEP, TimeUnit.EPOCH])
def test_step_and_epoch_intervals_are_never_locked(unit):
    # Locking an epoch interval to a step count would change where the segment ends: the epoch form
    # finishes after the epoch counter rolls over, the step form finishes inside the epoch. The two
    # counters would then disagree about which epoch the round ended in, and the next round would
    # replay an epoch with no batches left in it.
    schedule = SegmentSchedule(4, unit)
    progress = TrainProgress()
    schedule.begin(progress, now=0.0)
    for _ in range(4):
        progress.next_step(1)

    assert schedule.lock_to(progress) is False
    assert schedule.locked_target is None


def test_a_locked_target_records_the_full_progress_not_just_the_step_count():
    schedule = SegmentSchedule(1, TimeUnit.MINUTE)
    progress = TrainProgress(epoch=2, global_step=100)
    schedule.begin(progress, now=0.0)
    for _ in range(5):
        progress.next_step(1)
    progress.next_epoch()

    schedule.lock_to(progress)
    target = schedule.locked_target

    assert (target.epoch, target.epoch_step, target.global_step) == (3, 0, 105)


def test_never_is_not_a_usable_segment_unit():
    assert not SegmentSchedule.is_supported_unit(TimeUnit.NEVER)
    assert SegmentSchedule.is_supported_unit(TimeUnit.EPOCH)
    assert SegmentSchedule.is_supported_unit(TimeUnit.STEP)


def test_all_candidates_in_a_round_train_identical_step_counts():
    """Simulates the round loop to check the invariant the comparison depends on."""
    candidates = ["a", "b", "c"]
    schedule = SegmentSchedule(4, TimeUnit.STEP)
    shared = TrainProgress(epoch=0, global_step=0)

    for _round in range(3):
        schedule.unlock()
        steps_per_candidate = []
        round_start = TrainProgress(
            epoch=shared.epoch,
            epoch_step=shared.epoch_step,
            epoch_sample=shared.epoch_sample,
            global_step=shared.global_step,
        )

        for index, _candidate in enumerate(candidates):
            progress = TrainProgress(
                epoch=round_start.epoch,
                epoch_step=round_start.epoch_step,
                epoch_sample=round_start.epoch_sample,
                global_step=round_start.global_step,
            )
            schedule.begin(progress, now=0.0)
            while not schedule.is_complete(progress, now=0.0):
                progress.next_step(1)
            steps_per_candidate.append(schedule.steps_taken(progress))
            if index == 0:
                schedule.lock_to(progress)
                shared = progress  # the winner's progress is what the next round continues from

        assert steps_per_candidate == [4, 4, 4]

    assert shared.global_step == 12


# --- candidate merging -------------------------------------------------------------------------


def test_tunable_fields_come_from_the_candidate():
    base = TrainConfig.default_values()
    base.learning_rate = 1e-4
    base.mse_strength = 1.0
    base.optimizer.optimizer = Optimizer.ADAMW

    candidate = TrainConfig.default_values()
    candidate.learning_rate = 3e-4
    candidate.mse_strength = 0.5
    candidate.learning_rate_scheduler = LearningRateScheduler.COSINE
    candidate.optimizer.optimizer = Optimizer.PRODIGY

    merged_dict, ignored = merge_candidate_dict(base.to_dict(), candidate.to_dict())
    merged = TrainConfig.default_values().from_dict(merged_dict)

    assert merged.learning_rate == 3e-4
    assert merged.mse_strength == 0.5
    assert merged.learning_rate_scheduler == LearningRateScheduler.COSINE
    assert merged.optimizer.optimizer == Optimizer.PRODIGY
    assert ignored == []


def test_structural_fields_stay_on_the_base_and_are_reported():
    base = TrainConfig.default_values()
    base.lora_rank = 16
    base.batch_size = 4
    base.epochs = 20

    candidate = TrainConfig.default_values()
    candidate.lora_rank = 64
    candidate.batch_size = 1
    candidate.epochs = 20
    candidate.learning_rate = 5e-5

    merged_dict, ignored = merge_candidate_dict(base.to_dict(), candidate.to_dict())
    merged = TrainConfig.default_values().from_dict(merged_dict)

    assert merged.lora_rank == 16
    assert merged.batch_size == 4
    assert merged.learning_rate == 5e-5

    reported = " ".join(ignored)
    assert "lora_rank" in reported
    assert "batch_size" in reported
    assert "epochs" not in reported  # identical, so nothing to report


def test_model_part_learning_rate_is_tunable_but_the_train_flag_is_not():
    base = TrainConfig.default_values()
    base.unet.learning_rate = 1e-4
    base.unet.train = True

    candidate = TrainConfig.default_values()
    candidate.unet.learning_rate = 2e-4
    candidate.unet.train = False

    merged_dict, ignored = merge_candidate_dict(base.to_dict(), candidate.to_dict())
    merged = TrainConfig.default_values().from_dict(merged_dict)

    assert merged.unet.learning_rate == 2e-4
    assert merged.unet.train is True
    assert any("unet.train" in entry for entry in ignored)


def test_merging_does_not_mutate_the_base():
    base = TrainConfig.default_values()
    base.learning_rate = 1e-4
    base_dict = base.to_dict()
    snapshot = json.dumps(base_dict, sort_keys=True)

    candidate = TrainConfig.default_values()
    candidate.learning_rate = 9e-4
    candidate.unet.learning_rate = 7e-4

    merge_candidate_dict(base_dict, candidate.to_dict())

    assert json.dumps(base_dict, sort_keys=True) == snapshot


# --- loading candidates ------------------------------------------------------------------------


def test_load_candidate_configs_reads_every_filled_slot(tmp_path):
    base = make_base_config()
    base.multi_config_path_1 = write_config(tmp_path, "low_lr", learning_rate=1e-5)
    base.multi_config_path_3 = write_config(tmp_path, "high_lr", learning_rate=1e-3)

    candidates, errors = load_candidate_configs(base)

    assert errors == []
    assert [c.name for c in candidates] == ["low_lr", "high_lr"]
    assert candidates[0].config.learning_rate == 1e-5
    assert candidates[1].config.learning_rate == 1e-3
    # the slot index is preserved so log lines match what the user selected
    assert [c.index for c in candidates] == [0, 1]


def test_load_candidate_configs_reports_a_missing_file(tmp_path):
    base = make_base_config()
    base.multi_config_path_1 = write_config(tmp_path, "ok", learning_rate=1e-5)
    base.multi_config_path_2 = os.path.join(str(tmp_path), "does_not_exist.json")

    _candidates, errors = load_candidate_configs(base)

    assert len(errors) == 1
    assert "file not found" in errors[0]


def test_load_candidate_configs_reports_unparsable_json(tmp_path):
    broken = os.path.join(str(tmp_path), "broken.json")
    with open(broken, "w") as f:
        f.write("{not json")

    base = make_base_config()
    base.multi_config_path_1 = write_config(tmp_path, "ok", learning_rate=1e-5)
    base.multi_config_path_2 = broken

    _candidates, errors = load_candidate_configs(base)

    assert len(errors) == 1
    assert "could not read" in errors[0]


def test_candidates_with_the_same_basename_get_distinct_names(tmp_path):
    first_dir = os.path.join(str(tmp_path), "a")
    second_dir = os.path.join(str(tmp_path), "b")
    os.makedirs(first_dir)
    os.makedirs(second_dir)

    base = make_base_config()
    base.multi_config_path_1 = write_config(first_dir, "run", learning_rate=1e-5)
    base.multi_config_path_2 = write_config(second_dir, "run", learning_rate=1e-3)

    candidates, errors = load_candidate_configs(base)

    assert errors == []
    assert len({c.name for c in candidates}) == 2
    assert len({c.slug for c in candidates}) == 2


def test_candidate_config_inherits_the_base_workspace(tmp_path):
    base = make_base_config(workspace_dir="workspace/mine", cache_dir="cache/mine", epochs=7)
    base.multi_config_path_1 = write_config(
        tmp_path, "other", learning_rate=1e-5, workspace_dir="workspace/theirs", epochs=99
    )
    base.multi_config_path_2 = write_config(tmp_path, "another", learning_rate=1e-4)

    candidates, errors = load_candidate_configs(base)

    assert errors == []
    for candidate in candidates:
        assert candidate.config.workspace_dir == "workspace/mine"
        assert candidate.config.cache_dir == "cache/mine"
        assert candidate.config.epochs == 7
    assert any("workspace_dir" in entry for entry in candidates[0].ignored_fields)


# --- preflight ---------------------------------------------------------------------------------


def test_preflight_passes_for_a_valid_setup(tmp_path):
    base = make_base_config()
    base.multi_config_path_1 = write_config(tmp_path, "one", learning_rate=1e-5)
    base.multi_config_path_2 = write_config(tmp_path, "two", learning_rate=1e-4)

    assert preflight_errors(base) == []
    assert validate_before_start(base) == []


def test_preflight_is_silent_when_the_tournament_is_off():
    config = TrainConfig.default_values()
    config.multi_config = False
    config.validation = False
    assert preflight_errors(config) == []
    assert validate_before_start(config) == []


def test_preflight_requires_two_candidates(tmp_path):
    base = make_base_config()
    base.multi_config_path_1 = write_config(tmp_path, "one", learning_rate=1e-5)

    errors = preflight_errors(base)
    assert any("at least 2 config sets" in e for e in errors)


def test_preflight_requires_validation(tmp_path):
    base = make_base_config(validation=False)
    base.multi_config_path_1 = write_config(tmp_path, "one", learning_rate=1e-5)
    base.multi_config_path_2 = write_config(tmp_path, "two", learning_rate=1e-4)

    errors = preflight_errors(base)
    assert any("requires validation to be enabled" in e for e in errors)


def test_preflight_rejects_a_never_validation_interval(tmp_path):
    base = make_base_config(validate_after_unit=TimeUnit.NEVER)
    base.multi_config_path_1 = write_config(tmp_path, "one", learning_rate=1e-5)
    base.multi_config_path_2 = write_config(tmp_path, "two", learning_rate=1e-4)

    errors = preflight_errors(base)
    assert any("NEVER" in e for e in errors)


def test_preflight_rejects_a_duplicate_slot(tmp_path):
    shared = write_config(tmp_path, "one", learning_rate=1e-5)
    base = make_base_config()
    base.multi_config_path_1 = shared
    base.multi_config_path_2 = shared

    errors = preflight_errors(base)
    assert any("selected more than once" in e for e in errors)


def test_preflight_rejects_unsupported_modes(tmp_path):
    base = make_base_config(multi_gpu=True, only_cache=True)
    base.cloud.enabled = True
    base.multi_config_path_1 = write_config(tmp_path, "one", learning_rate=1e-5)
    base.multi_config_path_2 = write_config(tmp_path, "two", learning_rate=1e-4)

    errors = preflight_errors(base)
    assert any("multi-GPU" in e for e in errors)
    assert any("cloud" in e for e in errors)
    assert any("Only Cache" in e for e in errors)


def test_all_ten_slots_are_usable(tmp_path):
    base = make_base_config()
    base.multi_config_path_count = MULTI_CONFIG_SLOT_COUNT
    for slot in range(1, MULTI_CONFIG_SLOT_COUNT + 1):
        setattr(base, f"multi_config_path_{slot}", write_config(tmp_path, f"c{slot}", learning_rate=slot * 1e-5))

    candidates, errors = load_candidate_configs(base)
    assert errors == []
    assert len(candidates) == MULTI_CONFIG_SLOT_COUNT
    assert preflight_errors(base) == []


def test_slots_past_the_selected_count_are_ignored(tmp_path):
    base = make_base_config()
    base.multi_config_path_count = 2
    for slot in range(1, 5):
        setattr(base, f"multi_config_path_{slot}", write_config(tmp_path, f"c{slot}", learning_rate=slot * 1e-5))

    candidates, errors = load_candidate_configs(base)
    assert errors == []
    assert [c.name for c in candidates] == ["c1", "c2"]


def test_lowering_the_count_below_two_is_rejected(tmp_path):
    base = make_base_config()
    base.multi_config_path_count = 1
    base.multi_config_path_1 = write_config(tmp_path, "only", learning_rate=1e-5)

    assert any("at least 2 config sets" in e for e in preflight_errors(base))


# --- single-setting sweeps ------------------------------------------------------------------------


def make_sweep_config(setting: MultiConfigSweepSetting, *values: str, **overrides) -> TrainConfig:
    config = make_base_config(**overrides)
    config.multi_config_mode = MultiConfigMode.SINGLE_SETTING
    config.multi_config_sweep_setting = setting
    config.multi_config_sweep_count = len(values)
    for index, value in enumerate(values, start=1):
        setattr(config, f"multi_config_sweep_value_{index}", value)
    return config


def make_lora_sweep_config(setting: MultiConfigSweepSetting, *values: str, **overrides) -> TrainConfig:
    config = make_sweep_config(setting, *values, **overrides)
    config.model_type = ModelType.FLUX_DEV_1
    config.training_method = TrainingMethod.LORA
    return config


@pytest.mark.parametrize("setting", list(MultiConfigSweepSetting))
def test_every_sweep_setting_has_a_usable_registry_entry(setting):
    config = make_base_config()
    spec = SWEEP_SETTINGS[setting]

    assert spec.label
    assert spec.tooltip

    defaults = default_sweep_values(config, setting, 3)
    assert len(defaults) == 3
    for value in defaults:
        assert validate_sweep_value(config, setting, value) is None, f"{setting} suggested {value!r}"

    if spec.value_kind == SweepValueKind.CHOICE:
        assert sweep_options(config, setting), f"{setting} is a dropdown with no options"
    else:
        assert sweep_options(config, setting) == []


def test_learning_rate_sweep_varies_only_the_learning_rate():
    config = make_sweep_config(MultiConfigSweepSetting.LEARNING_RATE, "1e-5", "1e-4", "1e-3")
    config.optimizer.optimizer = Optimizer.PRODIGY
    config.epochs = 12

    candidates, errors = load_candidate_configs(config)

    assert errors == []
    assert [c.name for c in candidates] == ["1e-5", "1e-4", "1e-3"]
    assert [c.config.learning_rate for c in candidates] == [1e-5, 1e-4, 1e-3]
    for candidate in candidates:
        # everything else still comes from the training page
        assert candidate.config.optimizer.optimizer == Optimizer.PRODIGY
        assert candidate.config.epochs == 12
        assert not candidate.requires_model_rebuild


def test_optimizer_sweep_brings_each_optimizer_its_own_parameters():
    config = make_sweep_config(MultiConfigSweepSetting.OPTIMIZER, "ADAMW", "PRODIGY")

    candidates, errors = load_candidate_configs(config)

    assert errors == []
    assert [str(c.config.optimizer.optimizer) for c in candidates] == ["ADAMW", "PRODIGY"]
    # AdamW's default weight decay must not leak into Prodigy, which defaults to none
    assert candidates[0].config.optimizer.weight_decay == 1e-2
    assert candidates[1].config.optimizer.weight_decay == 0.0
    assert candidates[1].config.optimizer.d_coef == 1.0


def test_scheduler_and_timestep_distribution_sweeps_set_their_enums():
    config = make_sweep_config(MultiConfigSweepSetting.LEARNING_RATE_SCHEDULER, "CONSTANT", "COSINE")
    candidates, errors = load_candidate_configs(config)
    assert errors == []
    assert [c.config.learning_rate_scheduler for c in candidates] == [
        LearningRateScheduler.CONSTANT, LearningRateScheduler.COSINE
    ]

    config = make_sweep_config(MultiConfigSweepSetting.TIMESTEP_DISTRIBUTION, "UNIFORM", "LOGIT_NORMAL")
    candidates, errors = load_candidate_configs(config)
    assert errors == []
    assert [c.config.timestep_distribution for c in candidates] == [
        TimestepDistribution.UNIFORM, TimestepDistribution.LOGIT_NORMAL
    ]


def test_custom_scheduler_is_not_offered_because_it_needs_more_than_a_name():
    config = make_base_config()
    assert "CUSTOM" not in sweep_options(config, MultiConfigSweepSetting.LEARNING_RATE_SCHEDULER)
    assert validate_sweep_value(config, MultiConfigSweepSetting.LEARNING_RATE_SCHEDULER, "CUSTOM")


def test_layer_filter_sweep_resolves_presets_and_asks_for_a_rebuild():
    config = make_lora_sweep_config(MultiConfigSweepSetting.LAYER_FILTER, "attn-only", "full")

    candidates, errors = load_candidate_configs(config)

    assert errors == []
    assert candidates[0].config.layer_filter == "attn"
    assert candidates[0].config.layer_filter_preset == "attn-only"
    # "full" means every layer, which is expressed as an empty filter
    assert candidates[1].config.layer_filter == ""
    assert all(c.requires_model_rebuild for c in candidates)


def test_sweep_rejects_a_value_that_is_not_valid_for_the_setting():
    config = make_sweep_config(MultiConfigSweepSetting.OPTIMIZER, "ADAMW", "NOT_AN_OPTIMIZER")

    _candidates, errors = load_candidate_configs(config)
    assert any("NOT_AN_OPTIMIZER" in e for e in errors)
    assert any("NOT_AN_OPTIMIZER" in e for e in preflight_errors(config))


def test_sweep_rejects_a_learning_rate_that_is_not_a_number():
    config = make_sweep_config(MultiConfigSweepSetting.LEARNING_RATE, "1e-4", "fast")
    assert any("'fast' is not a number" in e for e in preflight_errors(config))


def test_sweep_rejects_a_non_positive_learning_rate():
    config = make_sweep_config(MultiConfigSweepSetting.LEARNING_RATE, "1e-4", "0")
    assert any("greater than zero" in e for e in preflight_errors(config))


def test_sweep_rejects_an_empty_value_slot():
    config = make_sweep_config(MultiConfigSweepSetting.LEARNING_RATE, "1e-4", "")
    assert any("no value selected" in e for e in preflight_errors(config))


def test_sweep_needs_at_least_two_values():
    config = make_sweep_config(MultiConfigSweepSetting.LEARNING_RATE, "1e-4")
    errors = preflight_errors(config)
    assert any("at least 2 values" in e for e in errors)


def test_sweep_rejects_repeated_values():
    config = make_sweep_config(MultiConfigSweepSetting.LEARNING_RATE, "1e-4", "1e-4")
    assert any("repeats" in e for e in preflight_errors(config))


def test_sweep_allows_the_full_ten_values():
    values = [f"{i}e-5" for i in range(1, MULTI_CONFIG_SWEEP_MAX + 1)]
    config = make_sweep_config(MultiConfigSweepSetting.LEARNING_RATE, *values)

    candidates, errors = load_candidate_configs(config)
    assert errors == []
    assert len(candidates) == MULTI_CONFIG_SWEEP_MAX
    assert preflight_errors(config) == []


def test_config_file_slots_are_ignored_in_single_setting_mode(tmp_path):
    config = make_sweep_config(MultiConfigSweepSetting.LEARNING_RATE, "1e-5", "1e-4")
    # a leftover, now missing, file selection must not block a sweep
    config.multi_config_path_1 = os.path.join(str(tmp_path), "gone.json")

    assert preflight_errors(config) == []
    candidates, errors = load_candidate_configs(config)
    assert errors == []
    assert [c.name for c in candidates] == ["1e-5", "1e-4"]


def test_layer_filter_sweep_requires_lora():
    config = make_lora_sweep_config(MultiConfigSweepSetting.LAYER_FILTER, "attn-only", "full")
    assert preflight_errors(config) == []

    config.training_method = TrainingMethod.FINE_TUNE
    assert any("only works with LoRA" in e for e in preflight_errors(config))


def test_layer_filter_sweep_rejects_ema():
    config = make_lora_sweep_config(MultiConfigSweepSetting.LAYER_FILTER, "attn-only", "full")
    config.ema = EMAMode.CPU
    assert any("EMA" in e for e in preflight_errors(config))


def test_other_sweeps_are_happy_with_ema():
    config = make_sweep_config(MultiConfigSweepSetting.LEARNING_RATE, "1e-5", "1e-4")
    config.ema = EMAMode.CPU
    assert preflight_errors(config) == []


# --- adaptive learning rate mode ------------------------------------------------------------------


def make_adaptive_config(start_lr: float, **overrides) -> TrainConfig:
    config = make_base_config(**overrides)
    config.multi_config_mode = MultiConfigMode.ADAPTIVE_LEARNING_RATE
    config.multi_config_adaptive_lr = start_lr
    return config


def test_adaptive_mode_builds_the_first_round_around_the_starting_rate():
    config = make_adaptive_config(0.0003)

    candidates, errors = load_candidate_configs(config)

    assert errors == []
    assert [c.name for c in candidates] == ["0.0002", "0.0003", "0.0004"]
    assert [c.config.learning_rate for c in candidates] == [0.0002, 0.0003, 0.0004]
    assert not any(c.requires_model_rebuild for c in candidates)


def test_adaptive_candidates_take_everything_else_from_the_base():
    config = make_adaptive_config(0.0003, epochs=9)
    config.optimizer.optimizer = Optimizer.PRODIGY

    candidates, _errors = load_candidate_configs(config)

    for candidate in candidates:
        assert candidate.config.epochs == 9
        assert candidate.config.optimizer.optimizer == Optimizer.PRODIGY


def test_build_ladder_candidates_follows_the_ladder():
    config = make_adaptive_config(0.0003)
    ladder = LearningRateLadder(0.0003)
    ladder.advance(0.0004)

    candidates = build_ladder_candidates(config, ladder)

    assert [c.name for c in candidates] == ["0.0003", "0.0004", "0.0005"]


def test_adaptive_mode_passes_preflight():
    assert preflight_errors(make_adaptive_config(0.0003)) == []
    assert validate_before_start(make_adaptive_config(0.0003)) == []


def test_adaptive_mode_rejects_a_non_positive_starting_rate():
    assert any("greater than zero" in e for e in preflight_errors(make_adaptive_config(0.0)))
    assert any("greater than zero" in e for e in preflight_errors(make_adaptive_config(-1e-4)))


def test_adaptive_mode_ignores_the_config_slots_and_sweep_values(tmp_path):
    config = make_adaptive_config(0.0003)
    config.multi_config_path_1 = os.path.join(str(tmp_path), "gone.json")
    config.multi_config_sweep_count = 4
    config.multi_config_sweep_value_1 = "nonsense"

    assert preflight_errors(config) == []


def test_end_early_is_off_by_default_and_independent_of_mode():
    config = make_base_config()
    assert config.multi_config_end_early is False

    config.multi_config_end_early = True
    config.multi_config_path_1 = "a"
    config.multi_config_path_2 = "b"
    # turning it on must not introduce validation errors of its own
    assert not any("end early" in e.lower() for e in preflight_errors(config))


# --- scoring and winner selection ---------------------------------------------------------------


def test_total_average_weights_by_sample_count():
    validation = result({"big": 1.0, "small": 3.0}, {"big": 9, "small": 1})
    assert score_validation_result(validation, MultiConfigSelection.TOTAL_AVERAGE) == pytest.approx(1.2)


def test_mean_per_concept_ignores_sample_count():
    validation = result({"big": 1.0, "small": 3.0}, {"big": 9, "small": 1})
    assert score_validation_result(validation, MultiConfigSelection.MEAN_PER_CONCEPT) == pytest.approx(2.0)


def test_worst_concept_takes_the_highest_loss():
    validation = result({"a": 0.2, "b": 0.9, "c": 0.4})
    assert score_validation_result(validation, MultiConfigSelection.WORST_CONCEPT) == pytest.approx(0.9)


def test_scoring_an_empty_result_gives_no_score():
    assert score_validation_result(None, MultiConfigSelection.TOTAL_AVERAGE) is None
    assert score_validation_result(ValidationResult(), MultiConfigSelection.TOTAL_AVERAGE) is None


def test_the_lowest_score_wins():
    assert pick_winner([0.5, 0.3, 0.9]) == 1


def test_candidates_without_a_score_are_skipped():
    assert pick_winner([None, 0.7, None]) == 1
    assert pick_winner([None, None]) is None
    assert pick_winner([]) is None


def test_a_tie_keeps_the_earlier_candidate():
    # not switching on an exact tie keeps the run on the settings it already has
    assert pick_winner([0.4, 0.4, 0.4]) == 0
