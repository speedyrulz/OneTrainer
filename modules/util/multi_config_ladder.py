"""The learning rate ladder used by the adaptive learning rate tournament.

The ladder is the sequence of learning rates made of a single leading digit, rolling over at each
decade:

    ... 0.00008  0.00009  0.0001  0.0002 ... 0.0009  0.001  0.002 ...

Every round trains a window of consecutive rungs, and the winner's position inside that window
decides where the window moves next. Winning in the middle leaves it where it is; winning below the
middle moves it down, above the middle moves it up, and the further from the middle the winner is the
further the window moves. With a window of 5 starting at 0.0001-0.0005, a win for 0.0004 (one above
the middle) shifts to 0.0002-0.0006, while a win for 0.0001 (two below) shifts to 0.00008-0.0003.

Rungs are numbered from the bottom of the ladder, so all of that is integer arithmetic on rung
indices and nothing has to be worked out in floating point.
"""

import math

# The ladder refuses to walk outside this range: below the floor a learning rate has stopped being
# training, and an unbounded walk would keep spending rounds on candidates that cannot win. The
# ceiling is 9.0 rather than something smaller because Prodigy and the D-Adaptation optimizers are
# used at a learning rate of 1.0 by convention, and the ladder has to be able to reach them.
MIN_EXPONENT = -12
MAX_EXPONENT = 0

# how many rungs a decade has: the leading digits 1..9
RUNGS_PER_DECADE = 9

# the number of learning rates a round may compare
MIN_WINDOW = 3
MAX_WINDOW = 10

MIN_INDEX = 0
MAX_INDEX = (MAX_EXPONENT - MIN_EXPONENT + 1) * RUNGS_PER_DECADE - 1


def decompose(value: float) -> tuple[int, int]:
    """Split a learning rate into its leading digit and its decade.

    ``0.0003`` becomes ``(3, -4)``. A value that is not a single leading digit is snapped to the
    nearest rung, because the ladder only has whole-digit steps to move by.
    """
    if value <= 0:
        raise ValueError(f"learning rate must be greater than zero, got {value}")

    exponent = math.floor(math.log10(value))
    mantissa = round(value / (10.0 ** exponent))

    # log10 is not exact for every power of ten, so the mantissa can land just outside 1..9
    if mantissa >= 10:
        mantissa = round(mantissa / 10)
        exponent += 1
    if mantissa < 1:
        mantissa = 1

    return int(mantissa), int(exponent)


def compose(mantissa: int, exponent: int) -> float:
    # built from text rather than by multiplying, so 9 * 10**-5 comes out as 9e-05 and not
    # 9.000000000000001e-05 - these values end up in filenames and log lines
    return float(f"{mantissa}e{exponent}")


def rung_index(value: float) -> int:
    """Where a learning rate sits on the ladder, counting up from the lowest rung."""
    mantissa, exponent = decompose(value)
    return (exponent - MIN_EXPONENT) * RUNGS_PER_DECADE + (mantissa - 1)


def rung_value(index: int) -> float | None:
    """The learning rate at a rung, or None past either end of the ladder."""
    if index < MIN_INDEX or index > MAX_INDEX:
        return None
    exponent, mantissa = divmod(index, RUNGS_PER_DECADE)
    return compose(mantissa + 1, exponent + MIN_EXPONENT)


def normalise(value: float) -> float:
    """The rung of the ladder a learning rate sits on."""
    return compose(*decompose(value))


def step(value: float, direction: int) -> float | None:
    """The next rung up (``direction`` 1) or down (``direction`` -1), or None past the ends."""
    return rung_value(rung_index(value) + direction)


def format_learning_rate(value: float) -> str:
    """A short, stable label. Used for candidate names, so it must round-trip through a filename."""
    return f"{value:g}"


def _round_away_from_zero(value: float) -> int:
    """Round halves outward, so an even-sized window always moves at least one rung.

    With an even count there is no middle rung, and the two candidates either side of the gap sit
    half a rung away from it. Rounding those to zero would leave the window stuck; rounding them
    outward is what makes "below the middle goes down, above the middle goes up" true for every
    position.
    """
    if value >= 0:
        return math.floor(value + 0.5)
    return math.ceil(value - 0.5)


class LearningRateLadder:
    """Tracks the window of rungs a round compares, and moves it towards the winner."""

    def __init__(self, start: float, count: int = MIN_WINDOW):
        self.count = max(MIN_WINDOW, min(int(count), MAX_WINDOW))

        centre = normalise(start)
        self.snapped_from: float | None = start if centre != start else None

        # the typed rate sits at the middle of the window, or just below it when the count is even
        self.low_index = self._clamp(rung_index(centre) - (self.count - 1) // 2)

    def _clamp(self, low_index: int) -> int:
        """Keep the whole window on the ladder, sliding it in from the ends rather than shrinking."""
        return max(MIN_INDEX, min(low_index, MAX_INDEX - self.count + 1))

    @property
    def centre(self) -> float:
        """The rate at the middle of the window, or just below it when the count is even."""
        return rung_value(self.low_index + (self.count - 1) // 2)

    def values(self) -> list[float]:
        """The learning rates to compare this round, lowest first."""
        return [rung_value(index) for index in range(self.low_index, self.low_index + self.count)]

    def advance(self, winner: float) -> bool:
        """Move the window according to where the winner sat in it. Returns whether it moved.

        The shift is the winner's distance from the middle of the window, so a narrow win nudges the
        window by one rung and a win at the edge moves it far enough that the winner is no longer at
        the edge next round.
        """
        position = rung_index(normalise(winner)) - self.low_index
        shift = _round_away_from_zero(position - (self.count - 1) / 2)

        new_low_index = self._clamp(self.low_index + shift)
        moved = new_low_index != self.low_index
        self.low_index = new_low_index
        return moved

    def describe(self) -> str:
        values = self.values()
        if len(values) <= 3:
            return " / ".join(format_learning_rate(value) for value in values)
        return (f"{format_learning_rate(values[0])} .. {format_learning_rate(values[-1])} "
                f"({len(values)} rates)")
