from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """The outcome of one validation pass.

    ``per_concept`` maps the concept label used in tensorboard to its average loss, and
    ``sample_counts`` maps the same label to the number of validation samples behind that average.
    ``total_average`` is the average over every sample, which is not the same as the mean of
    ``per_concept`` unless all concepts happen to have the same number of samples.
    """

    per_concept: dict[str, float] = field(default_factory=dict)
    sample_counts: dict[str, int] = field(default_factory=dict)
    total_average: float | None = None

    def total_samples(self) -> int:
        return sum(self.sample_counts.values())

    def is_empty(self) -> bool:
        return self.total_average is None or not self.per_concept

    def to_dict(self) -> dict:
        return {
            'per_concept': dict(self.per_concept),
            'sample_counts': dict(self.sample_counts),
            'total_average': self.total_average,
        }
