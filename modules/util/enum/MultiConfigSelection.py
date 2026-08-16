from enum import Enum


class MultiConfigSelection(Enum):
    # the average validation loss over every validation sample, regardless of concept
    TOTAL_AVERAGE = 'TOTAL_AVERAGE'

    # the unweighted mean of the per-concept average losses, so a small concept counts as much as a large one
    MEAN_PER_CONCEPT = 'MEAN_PER_CONCEPT'

    # the highest per-concept average loss, which picks the settings that leave no concept behind
    WORST_CONCEPT = 'WORST_CONCEPT'

    def __str__(self):
        return self.value
