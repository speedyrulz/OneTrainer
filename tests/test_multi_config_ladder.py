"""Tests for the adaptive learning rate ladder.

Run with:  pytest tests/test_multi_config_ladder.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.util.multi_config_ladder import (
    MAX_EXPONENT,
    MIN_EXPONENT,
    LearningRateLadder,
    compose,
    decompose,
    format_learning_rate,
    normalise,
    step,
)

import pytest

# --- the rungs ------------------------------------------------------------------------------------


@pytest.mark.parametrize(("value", "expected"), [
    (0.0003, (3, -4)),
    (0.001, (1, -3)),
    (0.0009, (9, -4)),
    (0.0001, (1, -4)),
    (9e-05, (9, -5)),
    (0.1, (1, -1)),
    (1e-8, (1, -8)),
])
def test_a_learning_rate_splits_into_leading_digit_and_decade(value, expected):
    assert decompose(value) == expected


@pytest.mark.parametrize(("mantissa", "exponent", "expected"), [
    (3, -4, 0.0003),
    (9, -5, 9e-05),
    (1, -3, 0.001),
])
def test_rungs_compose_back_to_clean_numbers(mantissa, exponent, expected):
    composed = compose(mantissa, exponent)
    assert composed == expected
    # built from text rather than by multiplying, so no float dust ends up in filenames
    assert format_learning_rate(composed) == f"{expected:g}"


def test_every_rung_survives_a_round_trip():
    for exponent in range(MIN_EXPONENT, MAX_EXPONENT + 1):
        for mantissa in range(1, 10):
            value = compose(mantissa, exponent)
            assert decompose(value) == (mantissa, exponent), f"{value} did not round-trip"


def test_a_value_between_rungs_snaps_to_the_nearest_one():
    assert normalise(3.4e-4) == 0.0003
    assert normalise(3.6e-4) == 0.0004


def test_a_learning_rate_of_zero_or_less_is_rejected():
    with pytest.raises(ValueError):
        decompose(0.0)
    with pytest.raises(ValueError):
        decompose(-1e-4)


# --- stepping -------------------------------------------------------------------------------------


def test_stepping_moves_the_leading_digit():
    assert step(0.0003, 1) == 0.0004
    assert step(0.0003, -1) == 0.0002


def test_stepping_rolls_over_at_a_decade():
    # the two boundaries called out when this was specified
    assert step(0.0001, -1) == 9e-05
    assert step(0.0009, 1) == 0.001


def test_stepping_up_and_back_down_returns_to_the_start():
    value = 0.0009
    assert step(step(value, 1), -1) == value

    value = 0.0001
    assert step(step(value, -1), 1) == value


def test_the_ladder_has_ends():
    assert step(compose(1, MIN_EXPONENT), -1) is None
    assert step(compose(9, MAX_EXPONENT), 1) is None


def test_walking_all_the_way_down_passes_through_every_rung():
    walked = []
    value = 0.001
    for _ in range(12):
        walked.append(format_learning_rate(value))
        value = step(value, -1)

    assert walked == [
        "0.001", "0.0009", "0.0008", "0.0007", "0.0006", "0.0005",
        "0.0004", "0.0003", "0.0002", "0.0001", "9e-05", "8e-05",
    ]


# --- the ladder -----------------------------------------------------------------------------------


def test_a_round_compares_the_centre_with_both_neighbours():
    ladder = LearningRateLadder(0.0003)
    assert ladder.values() == [0.0002, 0.0003, 0.0004]


def test_following_the_winner_upwards():
    ladder = LearningRateLadder(0.0003)
    assert ladder.advance(0.0004) is True
    assert ladder.values() == [0.0003, 0.0004, 0.0005]


def test_following_the_winner_downwards():
    ladder = LearningRateLadder(0.0003)
    assert ladder.advance(0.0002) is True
    assert ladder.values() == [0.0001, 0.0002, 0.0003]


def test_the_centre_winning_leaves_the_ladder_where_it_is():
    ladder = LearningRateLadder(0.0003)
    assert ladder.advance(0.0003) is False
    assert ladder.values() == [0.0002, 0.0003, 0.0004]


def test_the_ladder_crosses_a_decade_while_following_a_winner():
    ladder = LearningRateLadder(0.0002)
    ladder.advance(0.0001)
    assert ladder.values() == [9e-05, 0.0001, 0.0002]

    ladder.advance(9e-05)
    assert ladder.values() == [8e-05, 9e-05, 0.0001]


def test_a_ladder_at_the_very_end_slides_in_rather_than_shrinking():
    # the window always compares the requested number of rates; at the ends it slides inwards instead
    # of losing a candidate, so a round near the floor is still a real comparison
    ladder = LearningRateLadder(compose(1, MIN_EXPONENT))
    assert ladder.values() == [
        compose(1, MIN_EXPONENT), compose(2, MIN_EXPONENT), compose(3, MIN_EXPONENT)
    ]

    ladder = LearningRateLadder(compose(9, MAX_EXPONENT))
    assert ladder.values() == [
        compose(7, MAX_EXPONENT), compose(8, MAX_EXPONENT), compose(9, MAX_EXPONENT)
    ]


def test_the_window_cannot_be_pushed_off_the_ladder():
    ladder = LearningRateLadder(compose(2, MIN_EXPONENT))
    for _ in range(20):
        ladder.advance(ladder.values()[0])
        assert len(ladder.values()) == 3
        assert ladder.values()[0] == compose(1, MIN_EXPONENT)


def test_a_start_between_rungs_is_snapped_and_says_so():
    ladder = LearningRateLadder(3.4e-4)
    assert ladder.centre == 0.0003
    assert ladder.snapped_from == 3.4e-4

    ladder = LearningRateLadder(0.0003)
    assert ladder.snapped_from is None


# --- wider windows --------------------------------------------------------------------------------


def test_a_window_of_five_puts_the_start_in_the_middle():
    ladder = LearningRateLadder(0.0003, 5)
    assert ladder.values() == [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
    assert ladder.centre == 0.0003


def test_the_window_shifts_by_one_when_the_winner_is_one_off_the_middle():
    # the worked example: 0.0001-0.0005 with 0.0004 winning becomes 0.0002-0.0006
    ladder = LearningRateLadder(0.0003, 5)
    ladder.advance(0.0004)
    assert ladder.values() == [0.0002, 0.0003, 0.0004, 0.0005, 0.0006]


def test_the_window_shifts_by_two_when_the_winner_is_two_off_the_middle():
    # the other worked example: 0.0001 winning becomes 0.00008-0.0003
    ladder = LearningRateLadder(0.0003, 5)
    ladder.advance(0.0001)
    assert ladder.values() == [8e-05, 9e-05, 0.0001, 0.0002, 0.0003]


@pytest.mark.parametrize("count", [3, 5, 7, 9])
def test_an_odd_window_stays_put_when_its_middle_wins(count):
    ladder = LearningRateLadder(0.0005, count)
    before = ladder.values()
    assert ladder.advance(ladder.centre) is False
    assert ladder.values() == before


@pytest.mark.parametrize("count", [4, 6, 8, 10])
def test_an_even_window_always_moves(count):
    # there is no middle rung to win, so every position is either below or above the gap
    for position in range(count):
        ladder = LearningRateLadder(0.005, count)
        before = ladder.values()
        assert ladder.advance(before[position]) is True

        direction = -1 if position < count // 2 else 1
        moved = ladder.values()[0] - before[0]
        assert (moved > 0) == (direction > 0), \
            f"count {count} position {position} moved the wrong way"


@pytest.mark.parametrize("count", list(range(3, 11)))
def test_the_shift_grows_with_the_distance_from_the_middle(count):
    shifts = []
    for position in range(count):
        ladder = LearningRateLadder(0.005, count)
        low_before = ladder.low_index
        ladder.advance(ladder.values()[position])
        shifts.append(ladder.low_index - low_before)

    # monotonic across positions, and symmetric about the middle
    assert shifts == sorted(shifts)
    assert shifts[0] == -shifts[-1]
    assert shifts[0] <= -1 and shifts[-1] >= 1


@pytest.mark.parametrize("count", list(range(3, 11)))
def test_a_window_always_has_the_requested_number_of_rates(count):
    ladder = LearningRateLadder(0.0003, count)
    assert len(ladder.values()) == count
    assert len(set(ladder.values())) == count
    assert ladder.values() == sorted(ladder.values())


def test_the_window_count_is_clamped_to_what_the_ladder_supports():
    assert LearningRateLadder(0.0003, 1).count == 3
    assert LearningRateLadder(0.0003, 99).count == 10


def test_a_wide_window_reaches_a_distant_rate_faster_than_a_narrow_one():
    # the point of a wider window: an edge win moves further, so fewer rounds are needed
    narrow = LearningRateLadder(0.0003, 3)
    wide = LearningRateLadder(0.0003, 9)

    narrow_rounds = 0
    while narrow.values()[-1] < 0.005:
        narrow.advance(narrow.values()[-1])
        narrow_rounds += 1

    wide_rounds = 0
    while wide.values()[-1] < 0.005:
        wide.advance(wide.values()[-1])
        wide_rounds += 1

    assert wide_rounds < narrow_rounds


def test_a_long_walk_stays_on_the_ladder():
    # follow "lower always wins" for a while and check nothing drifts off the rungs
    ladder = LearningRateLadder(0.001)
    for _ in range(30):
        values = ladder.values()
        assert values == sorted(values)
        assert len(set(values)) == len(values), "a round must not compare the same rate twice"
        ladder.advance(values[0])
        assert normalise(ladder.centre) == ladder.centre
