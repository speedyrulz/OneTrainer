"""Support code for the multi-config validation tournament.

The tournament trains several candidate settings against each other in rounds. Every round starts
from one shared training state, runs each candidate for exactly one validation interval, validates,
and promotes the candidate with the best validation loss. The next round continues from the promoted
state, so the run greedily follows whichever settings are winning at that point in training.

Only a subset of settings may differ between candidates. Anything that changes the shape of the model,
the optimizer's parameter groups or the data pipeline has to stay fixed, because all candidates share
one loaded model and one cache. Those fields are listed in ``TUNABLE_*`` below; everything else is
taken from the base config (the settings currently open in the UI) and a difference in a candidate
file is reported but ignored.
"""

import json
import os
from copy import deepcopy
from dataclasses import dataclass, field

from modules.util import path_util
from modules.util.config.TrainConfig import (
    MULTI_CONFIG_MIN_CANDIDATES,
    MULTI_CONFIG_SWEEP_MAX,
    TrainConfig,
)
from modules.util.enum.EMAMode import EMAMode
from modules.util.enum.MultiConfigMode import MultiConfigMode
from modules.util.enum.MultiConfigSelection import MultiConfigSelection
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.multi_config_ladder import LearningRateLadder, format_learning_rate
from modules.util.multi_config_sweep import (
    apply_sweep_value,
    get_sweep_setting,
    validate_sweep_value,
)
from modules.util.TrainProgress import TrainProgress
from modules.util.ValidationResult import ValidationResult

# Top-level TrainConfig fields a candidate is allowed to override.
#
# A field belongs here only if it is read fresh out of self.config on every step (loss weights, noise
# settings, timestep sampling) or if it is only consumed when the optimizer / LR scheduler is built,
# both of which the tournament rebuilds at every candidate switch. Fields consumed while the model is
# constructed (LoRA rank, quantization, dtypes) or while the data pipeline is defined (masked_training,
# resolution, text encoder layer skip) are deliberately absent - changing those would need a full model
# reload and would invalidate the shared latent cache.
TUNABLE_FIELDS: frozenset[str] = frozenset({
    # learning rate and schedule
    "learning_rate",
    "learning_rate_scheduler",
    "custom_learning_rate_scheduler",
    "scheduler_params",
    "learning_rate_warmup_steps",
    "learning_rate_cycles",
    "learning_rate_min_factor",
    "learning_rate_scaler",
    "clip_grad_norm",

    # optimizer (the whole sub-config, including the optimizer type itself)
    "optimizer",

    # loss
    "mse_strength",
    "mae_strength",
    "log_cosh_strength",
    "huber_strength",
    "huber_delta",
    "vb_loss_strength",
    "loss_weight_fn",
    "loss_weight_strength",
    "loss_scaler",

    # noise and timestep sampling
    "offset_noise_weight",
    "generalized_offset_noise",
    "perturbation_noise_weight",
    "timestep_distribution",
    "min_noising_strength",
    "max_noising_strength",
    "noising_weight",
    "noising_bias",
    "timestep_shift",
    "dynamic_timestep_shifting",

    # masked training weights (masked_training itself is structural - it changes the data pipeline)
    "unmasked_probability",
    "unmasked_weight",
    "normalize_masked_area_loss",
    "masked_prior_preservation_weight",

    # embeddings
    "embedding_learning_rate",
    "preserve_embedding_norm",

    # EMA rate (EMAMode itself is structural - it decides whether an EMA model exists at all)
    "ema_decay",
    "ema_update_step_interval",
})

# Per-model-part fields a candidate is allowed to override. Only the learning rate: the remaining
# fields decide which parameters exist, how they are stored, or how the data pipeline is built.
TUNABLE_MODEL_PART_FIELDS: frozenset[str] = frozenset({
    "learning_rate",
})

# every TrainModelPartConfig attribute on TrainConfig
MODEL_PART_NAMES: tuple[str, ...] = (
    "unet",
    "prior",
    "transformer",
    "unconditional_transformer",
    "text_encoder",
    "text_encoder_2",
    "text_encoder_3",
    "text_encoder_4",
    "vae",
    "effnet_encoder",
    "decoder",
    "decoder_text_encoder",
    "decoder_vqgan",
)

# Structural fields worth telling the user about when a candidate file disagrees with the base config.
# Kept to settings a user would plausibly have meant to vary, so the report stays readable instead of
# listing every path that happens to differ between two saved configs.
REPORTED_STRUCTURAL_FIELDS: tuple[str, ...] = (
    "model_type",
    "training_method",
    "base_model_name",
    "lora_model_name",
    "peft_type",
    "lora_rank",
    "lora_alpha",
    "lora_decompose",
    "dropout_probability",
    "layer_filter",
    "layer_filter_preset",
    "epochs",
    "batch_size",
    "gradient_accumulation_steps",
    "resolution",
    "frames",
    "aspect_ratio_bucketing",
    "latent_caching",
    "masked_training",
    "train_dtype",
    "ema",
    "validation",
    "validate_after",
    "validate_after_unit",
    "text_encoder_layer_skip",
    "text_encoder_2_layer_skip",
    "rescale_noise_scheduler_to_zero_terminal_snr",
    "force_v_prediction",
    "force_epsilon_prediction",
    "concept_file_name",
    "workspace_dir",
    "cache_dir",
)

# per-part structural fields worth reporting
REPORTED_STRUCTURAL_PART_FIELDS: tuple[str, ...] = (
    "train",
    "weight_dtype",
    "stop_training_after",
    "stop_training_after_unit",
    "gradient_checkpointing",
    "offload_fraction",
)


@dataclass
class CandidateConfig:
    """One entry in the tournament: a name, where it came from, and the config to train with."""

    index: int
    name: str
    path: str
    config: TrainConfig
    ignored_fields: list[str] = field(default_factory=list)

    # True when this candidate trains a different set of weights than the others, so the model has to
    # be rebuilt when switching to it. Only a layer filter sweep sets this today.
    requires_model_rebuild: bool = False

    @property
    def slug(self) -> str:
        return path_util.safe_filename(self.name) or f"candidate{self.index + 1}"

    @property
    def label(self) -> str:
        return f"#{self.index + 1} {self.name}"


def candidate_name_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0] or path


def merge_candidate_dict(base_dict: dict, candidate_dict: dict) -> tuple[dict, list[str]]:
    """Overlay the tunable fields of ``candidate_dict`` onto ``base_dict``.

    Both dicts must already be in the current config schema. Returns the merged dict plus the names
    of the structural fields where the candidate disagreed with the base and was overruled.
    """
    merged = deepcopy(base_dict)
    ignored: list[str] = []

    for name in TUNABLE_FIELDS:
        if name in candidate_dict:
            merged[name] = deepcopy(candidate_dict[name])

    for part in MODEL_PART_NAMES:
        base_part = merged.get(part)
        candidate_part = candidate_dict.get(part)
        if not isinstance(base_part, dict) or not isinstance(candidate_part, dict):
            continue
        for name in TUNABLE_MODEL_PART_FIELDS:
            if name in candidate_part:
                base_part[name] = deepcopy(candidate_part[name])

    ignored.extend(
        f"{name}: {base_dict[name]!r} (candidate wanted {candidate_dict[name]!r})"
        for name in REPORTED_STRUCTURAL_FIELDS
        if name in candidate_dict and name in base_dict and candidate_dict[name] != base_dict[name]
    )

    for part in MODEL_PART_NAMES:
        base_part = base_dict.get(part)
        candidate_part = candidate_dict.get(part)
        if not isinstance(base_part, dict) or not isinstance(candidate_part, dict):
            continue
        ignored.extend(
            f"{part}.{name}: {base_part[name]!r} (candidate wanted {candidate_part[name]!r})"
            for name in REPORTED_STRUCTURAL_PART_FIELDS
            if name in candidate_part and name in base_part and candidate_part[name] != base_part[name]
        )

    return merged, ignored


def load_candidate_configs(base_config: TrainConfig) -> tuple[list[CandidateConfig], list[str]]:
    """Build the tournament's candidates, whichever mode it is in.

    Returns the candidates plus a list of hard errors. A candidate that fails to build produces an
    error and is left out, so the caller should refuse to start training if the error list is
    non-empty rather than silently running a shorter tournament.
    """
    if base_config.multi_config_mode == MultiConfigMode.ADAPTIVE_LEARNING_RATE:
        # the candidates depend on which learning rate is currently winning, so they are built per
        # round by the trainer rather than once up front
        return build_ladder_candidates(base_config, LearningRateLadder(base_config.multi_config_adaptive_lr, base_config.multi_config_adaptive_count)), []
    if base_config.multi_config_mode == MultiConfigMode.SINGLE_SETTING:
        return build_sweep_candidates(base_config)
    return load_candidate_config_files(base_config)


def build_ladder_candidates(
        base_config: TrainConfig,
        ladder: LearningRateLadder,
) -> list[CandidateConfig]:
    """One candidate per rung the ladder is currently comparing."""
    base_dict = base_config.to_dict()
    candidates = []

    for index, value in enumerate(ladder.values()):
        candidate_config = TrainConfig.default_values().from_dict(deepcopy(base_dict))
        candidate_config.learning_rate = value
        candidates.append(CandidateConfig(
            index=index,
            name=format_learning_rate(value),
            path=f"learning rate = {format_learning_rate(value)}",
            config=candidate_config,
        ))

    return candidates


def build_sweep_candidates(base_config: TrainConfig) -> tuple[list[CandidateConfig], list[str]]:
    """One candidate per swept value, each differing from the base config in exactly that setting."""
    candidates: list[CandidateConfig] = []
    errors: list[str] = []

    setting = base_config.multi_config_sweep_setting
    spec = get_sweep_setting(setting)
    base_dict = base_config.to_dict()

    for index, value in enumerate(base_config.multi_config_sweep_values()):
        error = validate_sweep_value(base_config, setting, value)
        if error is not None:
            errors.append(f"{spec.label} value #{index + 1}: {error}")
            continue

        candidate_config = TrainConfig.default_values().from_dict(deepcopy(base_dict))
        try:
            apply_sweep_value(candidate_config, setting, value)
        except Exception as e:
            errors.append(f"{spec.label} value #{index + 1}: could not apply {value!r}: {e}")
            continue

        candidates.append(CandidateConfig(
            index=index,
            name=value,
            path=f"{setting} = {value}",
            config=candidate_config,
            requires_model_rebuild=spec.rebuilds_model,
        ))

    return candidates, errors


def load_candidate_config_files(base_config: TrainConfig) -> tuple[list[CandidateConfig], list[str]]:
    """Read every filled-in config file slot and merge it onto ``base_config``."""
    candidates: list[CandidateConfig] = []
    errors: list[str] = []

    base_dict = base_config.to_dict()

    for index, raw_path in enumerate(base_config.multi_config_paths()):
        if not os.path.isfile(raw_path):
            errors.append(f"Config set #{index + 1}: file not found: {raw_path}")
            continue

        try:
            with open(raw_path, "r") as f:
                loaded_dict = json.load(f)
        except Exception as e:
            errors.append(f"Config set #{index + 1}: could not read {raw_path}: {e}")
            continue

        try:
            # from_dict migrates the file to the current schema, so old saved configs still work
            candidate_dict = TrainConfig.default_values().from_dict(loaded_dict).to_dict()
        except Exception as e:
            errors.append(f"Config set #{index + 1}: could not parse {raw_path}: {e}")
            continue

        merged_dict, ignored = merge_candidate_dict(base_dict, candidate_dict)

        try:
            merged_config = TrainConfig.default_values().from_dict(merged_dict)
        except Exception as e:
            errors.append(f"Config set #{index + 1}: could not build merged config from {raw_path}: {e}")
            continue

        candidates.append(CandidateConfig(
            index=index,
            name=candidate_name_from_path(raw_path),
            path=raw_path,
            config=merged_config,
            ignored_fields=ignored,
        ))

    duplicate_names = _duplicate_names([c.name for c in candidates])
    for candidate in candidates:
        if candidate.name in duplicate_names:
            # two files with the same basename would collide in filenames and log labels
            candidate.name = f"{candidate.name}({candidate.index + 1})"

    return candidates, errors


def _duplicate_names(names: list[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in names:
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    return duplicates


def preflight_errors(config: TrainConfig) -> list[str]:
    """Everything that must be true before a tournament can start, checked without loading a model."""
    errors: list[str] = []

    if not config.multi_config:
        return errors

    if config.multi_config_mode == MultiConfigMode.ADAPTIVE_LEARNING_RATE:
        errors.extend(_adaptive_lr_errors(config))
    elif config.multi_config_mode == MultiConfigMode.SINGLE_SETTING:
        errors.extend(_sweep_errors(config))
    else:
        errors.extend(_config_file_errors(config))

    if not config.validation:
        errors.append("Multi-config training requires validation to be enabled (general tab).")

    if not SegmentSchedule.is_supported_unit(config.validate_after_unit):
        errors.append("Multi-config training requires 'Validate after' to be something other than NEVER.")

    if config.validate_after_unit != TimeUnit.ALWAYS and config.validate_after <= 0:
        errors.append("Multi-config training requires 'Validate after' to be greater than zero.")

    if config.multi_gpu:
        errors.append("Multi-config training does not support multi-GPU training.")

    if config.cloud.enabled:
        errors.append("Multi-config training does not support cloud training.")

    if config.only_cache:
        errors.append("Multi-config training cannot be combined with 'Only Cache'.")

    return errors


def _config_file_errors(config: TrainConfig) -> list[str]:
    errors: list[str] = []

    paths = config.multi_config_paths()
    if len(paths) < MULTI_CONFIG_MIN_CANDIDATES:
        errors.append(
            f"Multi-config training needs at least {MULTI_CONFIG_MIN_CANDIDATES} config sets, "
            f"{len(paths)} selected."
        )

    seen: set[str] = set()
    for i, path in enumerate(paths):
        key = os.path.normcase(os.path.abspath(path))
        if key in seen:
            errors.append(f"Config set #{i + 1} is selected more than once: {path}")
        seen.add(key)
        if not os.path.isfile(path):
            errors.append(f"Config set #{i + 1} does not exist: {path}")

    return errors


def _adaptive_lr_errors(config: TrainConfig) -> list[str]:
    errors: list[str] = []

    start = config.multi_config_adaptive_lr
    if start is None or start <= 0:
        errors.append("The starting learning rate must be greater than zero.")
        return errors

    ladder = LearningRateLadder(start)
    if len(ladder.values()) < MULTI_CONFIG_MIN_CANDIDATES:
        errors.append(
            f"A starting learning rate of {format_learning_rate(start)} sits at the end of the "
            f"ladder, leaving nothing to compare it against."
        )

    return errors


def _sweep_errors(config: TrainConfig) -> list[str]:
    errors: list[str] = []

    setting = config.multi_config_sweep_setting
    spec = get_sweep_setting(setting)

    count = int(config.multi_config_sweep_count or 0)
    if count < MULTI_CONFIG_MIN_CANDIDATES:
        errors.append(
            f"A {spec.label} tournament needs at least {MULTI_CONFIG_MIN_CANDIDATES} values to have "
            f"anything to compare, {count} selected. Use one value only by setting it on the training "
            f"page and turning multi-config training off."
        )
    if count > MULTI_CONFIG_SWEEP_MAX:
        errors.append(f"At most {MULTI_CONFIG_SWEEP_MAX} values can be compared, {count} selected.")

    values = config.multi_config_sweep_values()
    seen: set[str] = set()
    for i, value in enumerate(values):
        error = validate_sweep_value(config, setting, value)
        if error is not None:
            errors.append(f"{spec.label} value #{i + 1}: {error}")
            continue
        if value in seen:
            errors.append(
                f"{spec.label} value #{i + 1} repeats {value!r}. Two identical candidates would train "
                f"the same thing twice."
            )
        seen.add(value)

    if spec.rebuilds_model:
        errors.extend(_model_rebuild_errors(config, spec.label))

    return errors


def _model_rebuild_errors(config: TrainConfig, label: str) -> list[str]:
    """Extra restrictions for a sweep that changes which weights are trained.

    Candidates no longer share a parameter set, so the promoted state can only be handed over by
    parameter name through the adapter. That rules out the cases where trainable state lives outside
    the adapter or is tracked positionally.
    """
    errors: list[str] = []

    if config.training_method != TrainingMethod.LORA:
        errors.append(
            f"A {label} tournament only works with LoRA training - the layer filter has no effect on "
            f"{config.training_method} training."
        )

    if config.ema != EMAMode.OFF:
        errors.append(
            f"A {label} tournament cannot be combined with EMA, because the EMA shadow weights are "
            f"tracked by position and every candidate trains a different set of weights."
        )

    if config.train_any_embedding() or config.train_any_output_embedding():
        errors.append(
            f"A {label} tournament cannot be combined with embedding training, because embedding "
            f"weights live outside the adapter that carries state between candidates."
        )

    return errors


def validate_before_start(config: TrainConfig) -> list[str]:
    """Everything the UI should block on, cheap enough to run on the button press.

    Returns an empty list when the tournament is disabled. The candidate files are actually parsed
    here so a broken file is reported in a dialog rather than as a traceback from the training thread.
    """
    errors = preflight_errors(config)
    if errors or not config.multi_config:
        return errors

    _, load_errors = load_candidate_configs(config)
    return load_errors


def score_validation_result(
        result: ValidationResult | None,
        selection: MultiConfigSelection,
) -> float | None:
    """Reduce a validation result to the single number the tournament ranks on. Lower is better."""
    if result is None or result.is_empty():
        return None

    match selection:
        case MultiConfigSelection.TOTAL_AVERAGE:
            return result.total_average
        case MultiConfigSelection.MEAN_PER_CONCEPT:
            losses = list(result.per_concept.values())
            return sum(losses) / len(losses) if losses else None
        case MultiConfigSelection.WORST_CONCEPT:
            losses = list(result.per_concept.values())
            return max(losses) if losses else None
        case _:
            return result.total_average


def pick_winner(scores: list[float | None]) -> int | None:
    """Index of the lowest score, ignoring candidates that produced no usable result.

    Ties go to the earliest candidate, which keeps the run on the settings it is already using
    instead of switching on noise.
    """
    best_index: int | None = None
    best_score: float | None = None
    for index, score in enumerate(scores):
        if score is None:
            continue
        if best_score is None or score < best_score:
            best_index = index
            best_score = score
    return best_index


class SegmentSchedule:
    """Decides when a candidate has trained for one full validation interval.

    Unlike the built-in validation trigger, which fires at the *start* of an interval (so the very
    first firing happens after a single step), this measures forward from wherever the segment
    started. A segment therefore always contains a complete interval of training, which is what makes
    two candidates comparable.
    """

    def __init__(self, interval: float, unit: TimeUnit):
        self.interval = interval
        self.unit = unit

        self._start_epoch = 0
        self._start_global_step = 0
        self._start_time = 0.0
        self._locked_target: TrainProgress | None = None

    @staticmethod
    def is_supported_unit(unit: TimeUnit) -> bool:
        return unit != TimeUnit.NEVER

    def uses_wall_clock(self) -> bool:
        return self.unit.is_time_unit()

    def begin(self, train_progress: TrainProgress, now: float) -> None:
        self._start_epoch = train_progress.epoch
        self._start_global_step = train_progress.global_step
        self._start_time = now

    def lock_to(self, train_progress: TrainProgress) -> bool:
        """Freeze the round to where the first candidate's segment ended. Returns whether it locked.

        Only wall-clock intervals need this. A step or epoch interval already ends at the same place
        for every candidate - they start from the same progress and consume the same batches - and
        locking those to a step count would be actively wrong: an epoch-length segment ends *after*
        the epoch counter rolls over, while a step-length one ends inside the epoch, so the two would
        disagree about which epoch the round finished in.

        A wall-clock interval has no such guarantee: a slower candidate would be judged after fewer
        steps. Locking makes every later candidate in the round stop at the first candidate's step
        count instead.
        """
        if not self.uses_wall_clock():
            return False
        self._locked_target = TrainProgress(
            epoch=train_progress.epoch,
            epoch_step=train_progress.epoch_step,
            epoch_sample=train_progress.epoch_sample,
            global_step=max(self._start_global_step + 1, train_progress.global_step),
        )
        return True

    def unlock(self) -> None:
        self._locked_target = None

    @property
    def locked_target(self) -> TrainProgress | None:
        """Where a locked round is expected to end, so the caller can normalise to it."""
        return self._locked_target

    def steps_taken(self, train_progress: TrainProgress) -> int:
        return train_progress.global_step - self._start_global_step

    def is_complete(self, train_progress: TrainProgress, now: float) -> bool:
        if self._locked_target is not None:
            return train_progress.global_step >= self._locked_target.global_step

        match self.unit:
            case TimeUnit.EPOCH:
                interval = max(1, int(self.interval))
                return train_progress.epoch >= self._start_epoch + interval
            case TimeUnit.STEP:
                interval = max(1, int(self.interval))
                return train_progress.global_step >= self._start_global_step + interval
            case TimeUnit.SECOND:
                return now - self._start_time >= self.interval and self.steps_taken(train_progress) >= 1
            case TimeUnit.MINUTE:
                return now - self._start_time >= self.interval * 60 and self.steps_taken(train_progress) >= 1
            case TimeUnit.HOUR:
                return now - self._start_time >= self.interval * 3600 and self.steps_taken(train_progress) >= 1
            case TimeUnit.ALWAYS:
                return self.steps_taken(train_progress) >= 1
            case _:
                return False

    def describe(self) -> str:
        if self._locked_target is not None:
            return f"{self._locked_target.global_step - self._start_global_step} step(s)"
        match self.unit:
            case TimeUnit.EPOCH:
                return f"{max(1, int(self.interval))} epoch(s)"
            case TimeUnit.STEP:
                return f"{max(1, int(self.interval))} step(s)"
            case TimeUnit.ALWAYS:
                return "1 step"
            case _:
                return f"{self.interval} {str(self.unit).lower()}(s)"
