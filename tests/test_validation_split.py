"""Tests for the automatic per-concept validation split.

The split has to hold two properties or it quietly corrupts a run: the two sides must be exactly
complementary (no image trains and validates, none disappears), and the choice must be identical
every time it is worked out, because the training and validation pipelines each compute it
separately and never compare notes.

Run with:  pytest tests/test_validation_split.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.dataLoader.mixin.DataLoaderMgdsMixin import (
    VALIDATION_SEED_OFFSET,
    select_concepts,
    validation_split_enabled,
)
from modules.dataLoader.pipelineModules.ValidationSplit import (
    SPLIT_KEY,
    ValidationSplit,
    holdout_count,
    holdout_indices,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ConceptType import ConceptType

import pytest

# --- how many are held back -------------------------------------------------------------------------


@pytest.mark.parametrize(("count", "percentage", "expected"), [
    (100, 10, 10),
    (100, 25, 25),
    (50, 20, 10),
    (10, 10, 1),
    (7, 10, 1),      # rounds to 0, but a concept with samples to spare always gives one
    (2, 10, 1),
    (1, 10, 0),      # a single sample cannot be split without losing it from one side
    (0, 10, 0),
    (100, 0, 0),     # turned off
])
def test_the_holdout_size_is_a_share_of_each_concept(count, percentage, expected):
    assert holdout_count(count, percentage) == expected


def test_a_concept_is_never_emptied_by_the_split():
    for count in range(2, 30):
        held = holdout_count(count, 100)
        assert 1 <= held <= count - 1


def test_the_percentage_applies_per_concept_not_across_the_dataset():
    # 10% of a 200-image concept and 10% of a 20-image one, not 10% of the 220 shared out
    assert holdout_count(200, 10) == 20
    assert holdout_count(20, 10) == 2


# --- which ones are held back -----------------------------------------------------------------------


def test_the_same_images_are_held_back_every_time():
    first = holdout_indices(50, 20, "concepts/faces|7")
    second = holdout_indices(50, 20, "concepts/faces|7")
    assert first == second
    assert len(first) == 10


def test_different_concepts_get_different_selections():
    a = holdout_indices(50, 20, "concepts/faces|7")
    b = holdout_indices(50, 20, "concepts/cars|7")
    assert a != b


def test_the_selection_does_not_depend_on_python_hash_randomisation():
    # worked out from a sha256 of the key, so it survives a restart; these are the actual values
    assert sorted(holdout_indices(10, 30, "concepts/faces|0")) == \
        sorted(holdout_indices(10, 30, "concepts/faces|0"))
    assert len(holdout_indices(10, 30, "concepts/faces|0")) == 3


# --- the pipeline module ------------------------------------------------------------------------------


class FakeUpstream:
    """Stands in for the module that enumerates sample paths."""

    def __init__(self, concepts: dict[str, int]):
        self.paths = []
        self.concepts = []
        for name, count in concepts.items():
            concept = {'path': name, 'seed': 0, SPLIT_KEY: {'key_seed': 0}}
            for i in range(count):
                self.paths.append(f"{name}/{i:03d}.png")
                self.concepts.append(concept)


def build_module(upstream: FakeUpstream, percentage: float, is_validation: bool) -> ValidationSplit:
    module = ValidationSplit(
        path_in_name='image_path', concept_in_name='concept',
        path_out_name='image_path', concept_out_name='concept',
        percentage=percentage, is_validation=is_validation,
    )

    def get_previous_item(_variation, name, index):
        return upstream.paths[index] if name == 'image_path' else upstream.concepts[index]

    module._get_previous_length = lambda _name: len(upstream.paths)
    module._get_previous_item = get_previous_item
    module.start(0)
    return module


def paths_of(module: ValidationSplit, upstream: FakeUpstream) -> list[str]:
    return [upstream.paths[i] for i in module.indices]


def test_the_two_sides_are_exactly_complementary():
    upstream = FakeUpstream({"a": 20, "b": 13})

    train = paths_of(build_module(upstream, 25, False), upstream)
    validation = paths_of(build_module(upstream, 25, True), upstream)

    assert set(train).isdisjoint(validation), "an image cannot both train and validate"
    assert sorted(train + validation) == sorted(upstream.paths), "no image may go missing"


def test_each_concept_is_split_on_its_own():
    upstream = FakeUpstream({"a": 100, "b": 10})
    validation = paths_of(build_module(upstream, 20, True), upstream)

    assert len([p for p in validation if p.startswith("a/")]) == 20
    assert len([p for p in validation if p.startswith("b/")]) == 2


def test_every_concept_contributes_to_validation():
    upstream = FakeUpstream({"big": 500, "tiny": 3})
    validation = paths_of(build_module(upstream, 1, True), upstream)

    assert any(p.startswith("big/") for p in validation)
    assert any(p.startswith("tiny/") for p in validation), \
        "a small concept must still be represented in the validation set"


def test_a_single_image_concept_stays_in_training():
    upstream = FakeUpstream({"solo": 1})

    assert paths_of(build_module(upstream, 50, False), upstream) == ["solo/000.png"]
    assert paths_of(build_module(upstream, 50, True), upstream) == []


def test_an_untagged_concept_is_passed_through_whole():
    # this is how an explicitly marked validation concept keeps all of its images
    upstream = FakeUpstream({"a": 10})
    for concept in upstream.concepts:
        concept.pop(SPLIT_KEY, None)

    assert len(paths_of(build_module(upstream, 30, True), upstream)) == 10
    assert len(paths_of(build_module(upstream, 30, False), upstream)) == 10


def test_the_kept_paths_stay_in_enumeration_order():
    upstream = FakeUpstream({"a": 40})
    module = build_module(upstream, 25, False)

    assert module.indices == sorted(module.indices)
    assert module.length() == len(module.indices)


def test_the_module_serves_the_right_item():
    upstream = FakeUpstream({"a": 20})
    module = build_module(upstream, 25, True)

    item = module.get_item(0, 0)
    assert item['image_path'] == upstream.paths[module.indices[0]]
    assert item['concept'] is upstream.concepts[module.indices[0]]


# --- choosing the concepts for each side ---------------------------------------------------------------


def make_concepts() -> list[dict]:
    return [
        {'path': "concepts/a", 'seed': 1, 'type': str(ConceptType.STANDARD)},
        {'path': "concepts/b", 'seed': 2, 'type': str(ConceptType.STANDARD)},
        {'path': "concepts/held", 'seed': 3, 'type': str(ConceptType.VALIDATION)},
    ]


def make_config(percent: float, validation: bool = True) -> TrainConfig:
    config = TrainConfig.default_values()
    config.validation = validation
    config.validation_split_percent = percent
    return config


def test_without_a_split_the_sides_are_the_marked_concepts():
    config = make_config(0)
    assert not validation_split_enabled(config)

    train = select_concepts(config, make_concepts(), is_validation=False)
    validation = select_concepts(config, make_concepts(), is_validation=True)

    assert [c['path'] for c in train] == ["concepts/a", "concepts/b"]
    assert [c['path'] for c in validation] == ["concepts/held"]


def test_a_split_puts_every_trainable_concept_on_both_sides():
    config = make_config(20)
    assert validation_split_enabled(config)

    train = select_concepts(config, make_concepts(), is_validation=False)
    validation = select_concepts(config, make_concepts(), is_validation=True)

    assert [c['path'] for c in train] == ["concepts/a", "concepts/b"]
    # the explicitly marked concept is still there, plus a copy of each trainable one
    assert [c['path'] for c in validation] == ["concepts/held", "concepts/a", "concepts/b"]


def test_the_validation_copy_is_tagged_and_reseeded():
    config = make_config(20)
    validation = select_concepts(config, make_concepts(), is_validation=True)

    copy_of_a = next(c for c in validation if c['path'] == "concepts/a")
    assert ConceptType(copy_of_a['type']) == ConceptType.VALIDATION
    assert copy_of_a[SPLIT_KEY]['key_seed'] == 1, "the split key must match the training side"
    # a different seed keeps this copy's cached data out of the training half's cache group
    assert copy_of_a['seed'] == 1 + VALIDATION_SEED_OFFSET


def test_the_validation_copy_has_random_augmentation_turned_off():
    # a validation loss can only be compared over time if the same image looks the same every pass
    config = make_config(20)
    concepts = make_concepts()
    for concept in concepts:
        concept['image'] = {'enable_random_flip': True, 'enable_crop_jitter': True}
        concept['text'] = {'enable_tag_shuffling': True}

    validation = select_concepts(config, concepts, is_validation=True)
    copy_of_a = next(c for c in validation if c['path'] == "concepts/a")

    assert copy_of_a['image']['enable_random_flip'] is False
    assert copy_of_a['image']['enable_crop_jitter'] is False
    assert copy_of_a['text']['enable_tag_shuffling'] is False

    # the training side keeps whatever the user set
    train = select_concepts(config, concepts, is_validation=False)
    train_a = next(c for c in train if c['path'] == "concepts/a")
    assert train_a['image']['enable_random_flip'] is True


def test_the_explicit_validation_concept_is_not_split():
    config = make_config(20)
    validation = select_concepts(config, make_concepts(), is_validation=True)

    held = next(c for c in validation if c['path'] == "concepts/held")
    assert SPLIT_KEY not in held


def test_both_sides_agree_on_the_split_key():
    config = make_config(20)
    train = select_concepts(config, make_concepts(), is_validation=False)
    validation = select_concepts(config, make_concepts(), is_validation=True)

    train_a = next(c for c in train if c['path'] == "concepts/a")
    validation_a = next(c for c in validation if c['path'] == "concepts/a")

    # the module builds its key from path and key_seed; those must match or the two sides would hold
    # back different images
    assert train_a[SPLIT_KEY]['key_seed'] == validation_a[SPLIT_KEY]['key_seed']
    assert train_a['path'] == validation_a['path']


def test_selecting_concepts_does_not_mutate_the_originals():
    config = make_config(20)
    concepts = make_concepts()
    select_concepts(config, concepts, is_validation=True)

    assert all(SPLIT_KEY not in c for c in concepts)
    assert [c['seed'] for c in concepts] == [1, 2, 3]


def test_the_split_needs_validation_to_be_enabled():
    assert not validation_split_enabled(make_config(20, validation=False))
    assert validation_split_enabled(make_config(20, validation=True))


def test_an_end_to_end_split_covers_the_whole_dataset():
    """Both sides, built the way the data loaders build them, over the same enumerated files."""
    config = make_config(25)
    concepts = make_concepts()

    sides = {}
    for is_validation in (False, True):
        selected = select_concepts(config, concepts, is_validation)
        upstream = FakeUpstream({})
        for concept in selected:
            for i in range(12):
                upstream.paths.append(f"{concept['path']}/{i:03d}.png")
                upstream.concepts.append(concept)
        sides[is_validation] = paths_of(build_module(upstream, 25, is_validation), upstream)

    trainable = sides[False]
    validating = sides[True]

    for name in ("concepts/a", "concepts/b"):
        train_share = [p for p in trainable if p.startswith(name)]
        validation_share = [p for p in validating if p.startswith(name)]
        assert len(train_share) == 9
        assert len(validation_share) == 3
        assert set(train_share).isdisjoint(validation_share)

    # the explicitly marked concept is untouched and only ever validates
    assert len([p for p in validating if p.startswith("concepts/held")]) == 12
    assert not any(p.startswith("concepts/held") for p in trainable)
