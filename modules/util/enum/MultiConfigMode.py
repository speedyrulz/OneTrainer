from enum import Enum


class MultiConfigMode(Enum):
    # race up to five saved config files against each other. Each may differ from the base config in
    # any of the tunable settings at once.
    FULL_CONFIGS = 'FULL_CONFIGS'

    # race up to ten values of a single setting against each other. Everything else comes from the
    # training page, so the comparison isolates one variable.
    SINGLE_SETTING = 'SINGLE_SETTING'

    # race a learning rate against its two neighbours on the ladder, and follow the winner. The
    # candidates change every round, so the learning rate walks up and down during the run instead of
    # being decided once.
    ADAPTIVE_LEARNING_RATE = 'ADAPTIVE_LEARNING_RATE'

    def __str__(self):
        return self.value
