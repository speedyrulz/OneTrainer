from enum import Enum


class MultiConfigPruneStrategy(Enum):
    # drop every config that won none of the rounds in the window. Clears out several at once, but a
    # config that is consistently second by a hair is dropped just as readily as one that is nowhere
    # near.
    NEVER_WON = 'NEVER_WON'

    # drop the single config with the highest average validation loss over the window. Slower to
    # narrow the field, but it ranks on how far behind a config actually is rather than on whether it
    # happened to come first.
    WORST_AVERAGE = 'WORST_AVERAGE'

    def __str__(self):
        return self.value
