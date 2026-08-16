from enum import Enum


class MultiConfigSweepSetting(Enum):
    """The training setting a single-setting tournament varies.

    To add another: add a member here and one entry in ``modules.util.multi_config_sweep.SWEEP_SETTINGS``.
    Nothing else needs to change - the UI builds its inputs from the registry, and candidate
    generation reads it too.
    """

    OPTIMIZER = 'OPTIMIZER'
    LEARNING_RATE_SCHEDULER = 'LEARNING_RATE_SCHEDULER'
    LEARNING_RATE = 'LEARNING_RATE'
    TIMESTEP_DISTRIBUTION = 'TIMESTEP_DISTRIBUTION'
    LAYER_FILTER = 'LAYER_FILTER'

    def __str__(self):
        return self.value
