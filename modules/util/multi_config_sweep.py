"""The settings a single-setting tournament can vary, and how to apply them.

Everything about a sweepable setting lives in one ``SweepSetting`` entry: what the user picks from,
how a picked value is written onto a config, how it is labelled, and whether changing it forces the
model to be rebuilt. The UI builds its inputs from this registry and the trainer reads it too, so
adding a setting means adding one member to
:class:`~modules.util.enum.MultiConfigSweepSetting.MultiConfigSweepSetting` and one entry to
``SWEEP_SETTINGS`` below.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.LearningRateScheduler import LearningRateScheduler
from modules.util.enum.MultiConfigSweepSetting import MultiConfigSweepSetting
from modules.util.enum.Optimizer import Optimizer
from modules.util.enum.TimestepDistribution import TimestepDistribution


class SweepValueKind(Enum):
    # the user picks from a fixed list, shown as a dropdown
    CHOICE = 'CHOICE'

    # the user types a number, shown as a text box
    NUMBER = 'NUMBER'


@dataclass(frozen=True)
class SweepSetting:
    label: str
    tooltip: str
    value_kind: SweepValueKind

    # the values the user may pick, given the current config. Empty for NUMBER settings.
    options: Callable[[TrainConfig], list[str]]

    # write one value onto a config. Only called with a value that passed validate.
    apply: Callable[[TrainConfig, str], None]

    # None if the value is usable, otherwise the reason it is not
    validate: Callable[[TrainConfig, str], str | None]

    # a sensible starting point when the user first opens the tab
    default_values: Callable[[TrainConfig], list[str]]

    # True when changing this setting changes which parameters are trained, which means the model has
    # to be rebuilt between candidates and optimizer state cannot carry across a change
    rebuilds_model: bool = False


def _enum_names(enum_cls) -> list[str]:
    return [str(member) for member in enum_cls]


def _validate_choice(options: Callable[[TrainConfig], list[str]], what: str):
    def validate(config: TrainConfig, value: str) -> str | None:
        available = options(config)
        if value not in available:
            return f"{value!r} is not a valid {what}"
        return None

    return validate


def _apply_optimizer(config: TrainConfig, value: str) -> None:
    # Import here: optimizer_util imports TrainConfig, and this module is imported from the UI layer
    # that TrainConfig has no business knowing about.
    from modules.util.optimizer_util import change_optimizer

    config.optimizer.optimizer = Optimizer[value]
    # Swap in that optimizer's own parameters rather than leaving the previous optimizer's behind -
    # betas and epsilon mean different things to different optimizers, and a stale weight decay would
    # quietly change what is being compared. Any settings the user saved for this optimizer win.
    config.optimizer.from_dict(change_optimizer(config).to_dict())


def _validate_learning_rate(_config: TrainConfig, value: str) -> str | None:
    try:
        parsed = float(value)
    except ValueError:
        return f"{value!r} is not a number"
    if parsed <= 0:
        return f"learning rate must be greater than zero, got {value}"
    return None


def _apply_learning_rate(config: TrainConfig, value: str) -> None:
    config.learning_rate = float(value)


def layer_presets(config: TrainConfig) -> dict:
    """The layer filter presets the current model type and training method offer."""
    from modules.util import create

    cls = create.get_model_setup_class(config.model_type, config.training_method)
    presets = getattr(cls, "LAYER_PRESETS", None) if cls is not None else None
    return presets if presets else {"full": []}


def _layer_preset_patterns(preset_definition) -> tuple[str, bool]:
    # a preset is either a bare pattern list or {"patterns": [...], "regex": bool}, matching what the
    # layer filter control on the training page accepts
    if isinstance(preset_definition, dict):
        return ",".join(preset_definition.get("patterns", [])), bool(preset_definition.get("regex", False))
    return ",".join(preset_definition or []), False


def _apply_layer_filter(config: TrainConfig, value: str) -> None:
    patterns, uses_regex = _layer_preset_patterns(layer_presets(config).get(value, []))
    config.layer_filter_preset = value
    config.layer_filter = patterns
    config.layer_filter_regex = uses_regex


SWEEP_SETTINGS: dict[MultiConfigSweepSetting, SweepSetting] = {
    MultiConfigSweepSetting.OPTIMIZER: SweepSetting(
        label="Optimizer",
        tooltip="Race optimizers against each other. Each value also brings that optimizer's own "
                "parameters, including any you saved for it in the optimizer settings window.",
        value_kind=SweepValueKind.CHOICE,
        options=lambda _config: _enum_names(Optimizer),
        apply=_apply_optimizer,
        validate=_validate_choice(lambda _config: _enum_names(Optimizer), "optimizer"),
        default_values=lambda config: _first_choices(_enum_names(Optimizer), str(config.optimizer.optimizer)),
    ),
    MultiConfigSweepSetting.LEARNING_RATE_SCHEDULER: SweepSetting(
        label="Learning Rate Scheduler",
        tooltip="Race learning rate schedules against each other. CUSTOM is excluded, because it "
                "needs a class name and parameters that cannot be varied from a dropdown.",
        value_kind=SweepValueKind.CHOICE,
        options=lambda _config: [
            name for name in _enum_names(LearningRateScheduler) if name != str(LearningRateScheduler.CUSTOM)
        ],
        apply=lambda config, value: setattr(config, "learning_rate_scheduler", LearningRateScheduler[value]),
        validate=_validate_choice(
            lambda _config: [
                name for name in _enum_names(LearningRateScheduler) if name != str(LearningRateScheduler.CUSTOM)
            ],
            "learning rate scheduler",
        ),
        default_values=lambda config: _first_choices(
            [name for name in _enum_names(LearningRateScheduler) if name != str(LearningRateScheduler.CUSTOM)],
            str(config.learning_rate_scheduler),
        ),
    ),
    MultiConfigSweepSetting.LEARNING_RATE: SweepSetting(
        label="Learning Rate",
        tooltip="Race learning rates against each other. Type one value per box, for example 1e-4.",
        value_kind=SweepValueKind.NUMBER,
        options=lambda _config: [],
        apply=_apply_learning_rate,
        validate=_validate_learning_rate,
        default_values=lambda config: _learning_rate_ladder(config.learning_rate),
    ),
    MultiConfigSweepSetting.TIMESTEP_DISTRIBUTION: SweepSetting(
        label="Timestep Distribution",
        tooltip="Race timestep distributions against each other. The distribution's own weight and "
                "bias values come from the training page and stay the same for every candidate.",
        value_kind=SweepValueKind.CHOICE,
        options=lambda _config: _enum_names(TimestepDistribution),
        apply=lambda config, value: setattr(config, "timestep_distribution", TimestepDistribution[value]),
        validate=_validate_choice(lambda _config: _enum_names(TimestepDistribution), "timestep distribution"),
        default_values=lambda config: _first_choices(
            _enum_names(TimestepDistribution), str(config.timestep_distribution)
        ),
    ),
    MultiConfigSweepSetting.LAYER_FILTER: SweepSetting(
        label="Layer Filter",
        tooltip="Race layer filter presets against each other, to find which part of the model is "
                "worth training. This is the one setting that changes which weights exist, so the "
                "adapter is rebuilt between candidates and optimizer state does not carry across.",
        value_kind=SweepValueKind.CHOICE,
        options=lambda config: list(layer_presets(config).keys()),
        apply=_apply_layer_filter,
        validate=_validate_choice(lambda config: list(layer_presets(config).keys()), "layer filter preset"),
        default_values=lambda config: _first_choices(
            list(layer_presets(config).keys()), config.layer_filter_preset
        ),
        rebuilds_model=True,
    ),
}


def _first_choices(available: list[str], current: str) -> list[str]:
    """A starting list that leads with whatever the training page is set to."""
    if not available:
        return []
    if current not in available:
        return list(available)
    return [current] + [name for name in available if name != current]


def _learning_rate_ladder(current: float) -> list[str]:
    """A half-order-of-magnitude ladder around the current learning rate."""
    base = current if current and current > 0 else 1e-4
    return [f"{base * factor:g}" for factor in (0.3, 1.0, 3.0)]


def get_sweep_setting(setting: MultiConfigSweepSetting) -> SweepSetting:
    return SWEEP_SETTINGS[setting]


def sweep_options(config: TrainConfig, setting: MultiConfigSweepSetting) -> list[str]:
    return get_sweep_setting(setting).options(config)


def apply_sweep_value(config: TrainConfig, setting: MultiConfigSweepSetting, value: str) -> None:
    get_sweep_setting(setting).apply(config, value)


def validate_sweep_value(config: TrainConfig, setting: MultiConfigSweepSetting, value: str) -> str | None:
    if not value:
        return "no value selected"
    return get_sweep_setting(setting).validate(config, value)


def default_sweep_values(config: TrainConfig, setting: MultiConfigSweepSetting, count: int) -> list[str]:
    """``count`` starting values to pre-fill empty slots with.

    Cycles rather than repeating the last value, so the suggestions stay distinct for as long as the
    setting has choices left. Asking for more values than a setting has produces duplicates, which
    the pre-flight check reports rather than silently training the same thing twice.
    """
    suggested = get_sweep_setting(setting).default_values(config)
    if not suggested:
        return [""] * count
    return [suggested[i % len(suggested)] for i in range(count)]


def sweep_rebuilds_model(setting: MultiConfigSweepSetting) -> bool:
    return get_sweep_setting(setting).rebuilds_model
