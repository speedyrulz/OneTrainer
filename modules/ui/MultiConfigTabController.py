from modules.util.config.TrainConfig import MULTI_CONFIG_SLOT_COUNT, MULTI_CONFIG_SWEEP_MAX, TrainConfig
from modules.util.enum.MultiConfigMode import MultiConfigMode
from modules.util.enum.MultiConfigPruneStrategy import MultiConfigPruneStrategy
from modules.util.enum.MultiConfigSweepSetting import MultiConfigSweepSetting
from modules.util.multi_config_ladder import MAX_WINDOW, MIN_WINDOW, LearningRateLadder
from modules.util.multi_config_sweep import (
    SWEEP_SETTINGS,
    SweepValueKind,
    default_sweep_values,
    get_sweep_setting,
    sweep_options,
    validate_sweep_value,
)


class MultiConfigTabController:
    def __init__(self, config: TrainConfig):
        self.config = config

    def get_modes(self) -> list[tuple[str, MultiConfigMode]]:
        return [
            ("Full configs", MultiConfigMode.FULL_CONFIGS),
            ("Single setting", MultiConfigMode.SINGLE_SETTING),
            ("Adaptive learning rate", MultiConfigMode.ADAPTIVE_LEARNING_RATE),
        ]

    def ladder_preview(self) -> str:
        """What the first round would compare, or why it cannot be worked out yet.

        Recomputed as the user types, so it has to cope with half-finished numbers.
        """
        try:
            ladder = LearningRateLadder(
                self.config.multi_config_adaptive_lr, self.config.multi_config_adaptive_count
            )
        except (ValueError, TypeError):
            return "enter a learning rate greater than zero"

        values = ladder.values()
        preview = " / ".join(f"{value:g}" for value in values)
        if ladder.snapped_from is not None:
            preview += f"   (snapped from {ladder.snapped_from:g})"
        return preview

    def get_sweep_settings(self) -> list[tuple[str, MultiConfigSweepSetting]]:
        return [(spec.label, setting) for setting, spec in SWEEP_SETTINGS.items()]

    def get_sweep_counts(self) -> list[str]:
        return [str(i) for i in range(1, MULTI_CONFIG_SWEEP_MAX + 1)]

    def get_path_counts(self) -> list[str]:
        return [str(i) for i in range(1, MULTI_CONFIG_SLOT_COUNT + 1)]

    def get_prune_strategies(self) -> list[tuple[str, MultiConfigPruneStrategy]]:
        return [
            ("Never won", MultiConfigPruneStrategy.NEVER_WON),
            ("Worst average loss", MultiConfigPruneStrategy.WORST_AVERAGE),
        ]

    def get_adaptive_counts(self) -> list[str]:
        return [str(i) for i in range(MIN_WINDOW, MAX_WINDOW + 1)]

    @property
    def path_count(self) -> int:
        return max(1, min(int(self.config.multi_config_path_count or 1), MULTI_CONFIG_SLOT_COUNT))

    @property
    def sweep_setting(self) -> MultiConfigSweepSetting:
        return self.config.multi_config_sweep_setting

    @property
    def sweep_count(self) -> int:
        return max(1, min(int(self.config.multi_config_sweep_count or 1), MULTI_CONFIG_SWEEP_MAX))

    def sweep_label(self) -> str:
        return get_sweep_setting(self.sweep_setting).label

    def sweep_tooltip(self) -> str:
        return get_sweep_setting(self.sweep_setting).tooltip

    def sweep_is_choice(self) -> bool:
        return get_sweep_setting(self.sweep_setting).value_kind == SweepValueKind.CHOICE

    def sweep_choices(self) -> list[str]:
        return sweep_options(self.config, self.sweep_setting)

    def sweep_notice(self) -> str | None:
        """A warning to show under the setting picker, or None when there is nothing to say."""
        if get_sweep_setting(self.sweep_setting).rebuilds_model:
            return (
                "This setting changes which weights are trained, so the values cannot hand their "
                "progress to each other. Each value trains its own model and the rounds report which "
                "one is ahead. Needs LoRA training with EMA and embedding training off."
            )
        return None

    def repair_sweep_values(self) -> dict[str, str]:
        """Values for any visible slot whose contents do not fit the selected setting.

        Switching from Learning Rate to Optimizer leaves numbers in boxes that now want optimizer
        names, and raising the value count reveals empty boxes. Both get a usable starting value.
        Slots that already hold something valid are left alone, so switching settings back and forth
        keeps whatever the user typed for each of them. Slots beyond the current count are not
        touched: they are not on screen, and filling them would discard values the user may come back
        to by raising the count again.
        """
        setting = self.sweep_setting
        defaults = default_sweep_values(self.config, setting, MULTI_CONFIG_SWEEP_MAX)

        repaired: dict[str, str] = {}
        for slot in range(1, self.sweep_count + 1):
            name = f"multi_config_sweep_value_{slot}"
            current = (getattr(self.config, name, "") or "").strip()
            if validate_sweep_value(self.config, setting, current) is not None:
                repaired[name] = defaults[slot - 1]
        return repaired
