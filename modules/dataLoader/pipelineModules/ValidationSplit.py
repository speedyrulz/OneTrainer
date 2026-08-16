import hashlib
import random

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule

# key added to a concept dict by the data loader to say which side of the split this pipeline wants.
# A concept without it is passed through untouched, which is how an explicitly marked validation
# concept keeps all of its images.
SPLIT_KEY = "validation_split"


def holdout_count(sample_count: int, percentage: float) -> int:
    """How many of a concept's samples are held back for validation.

    Always at least one, so a small concept still contributes something to validate against, and
    always at least one short of the whole concept, so nothing is left with no training data. A
    concept with a single sample is never split - there is no way to divide it without losing it from
    one side or the other.
    """
    if percentage <= 0 or sample_count < 2:
        return 0
    return max(1, min(round(sample_count * percentage / 100.0), sample_count - 1))


def holdout_indices(sample_count: int, percentage: float, key: str) -> set[int]:
    """Which of a concept's samples are held back, chosen at random but always the same ones.

    The training and validation pipelines each work this out for themselves, so the choice has to
    depend only on the concept and not on anything about the run. It is seeded from a hash of the
    concept key rather than the built-in ``hash``, which is salted per process and would put an image
    in the training set one run and the validation set the next.
    """
    count = holdout_count(sample_count, percentage)
    if count == 0:
        return set()

    digest = hashlib.sha256(key.encode("utf-8")).digest()
    generator = random.Random(int.from_bytes(digest[:8], "big"))
    return set(generator.sample(range(sample_count), count))


class ValidationSplit(
    PipelineModule,
    RandomAccessPipelineModule,
):
    """Keeps one side of an automatic per-concept validation split.

    Sits directly after the module that enumerates sample paths and drops the ones belonging to the
    other side, so everything downstream sees a smaller dataset and needs no changes. Both pipelines
    run the same module over the same enumerated paths and pick opposite halves, which is what makes
    the two sides exactly complementary.
    """

    def __init__(
            self,
            path_in_name: str,
            concept_in_name: str,
            path_out_name: str,
            concept_out_name: str,
            percentage: float,
            is_validation: bool,
    ):
        super().__init__()

        self.path_in_name = path_in_name
        self.concept_in_name = concept_in_name
        self.path_out_name = path_out_name
        self.concept_out_name = concept_out_name
        self.percentage = percentage
        self.is_validation = is_validation

        self.indices: list[int] = []

    def length(self) -> int:
        return len(self.indices)

    def get_inputs(self) -> list[str]:
        return [self.path_in_name, self.concept_in_name]

    def get_outputs(self) -> list[str]:
        return [self.path_out_name, self.concept_out_name]

    def start(self, variation: int):
        groups: dict[str, list[int]] = {}
        concepts: dict[str, dict] = {}

        for index in range(self._get_previous_length(self.path_in_name)):
            concept = self._get_previous_item(variation, self.concept_in_name, index)
            key = self.__concept_key(concept)
            groups.setdefault(key, []).append(index)
            concepts[key] = concept

        kept: list[int] = []

        for key, indices in groups.items():
            split = concepts[key].get(SPLIT_KEY)
            if not split:
                # not part of the split: an explicitly marked validation concept, or a concept the
                # split does not apply to. Everything it has belongs to this pipeline.
                kept.extend(indices)
                continue

            holdout = holdout_indices(len(indices), self.percentage, key)
            for position, index in enumerate(indices):
                if (position in holdout) == self.is_validation:
                    kept.append(index)

        # back into enumeration order, so the dataset keeps the ordering the pipeline gave it
        self.indices = sorted(kept)

    @staticmethod
    def __concept_key(concept: dict) -> str:
        # identifies the concept the split is computed for. The seed is included so two concepts
        # pointing at the same directory still get their own split, but the validation clone's seed is
        # deliberately left out of it - both sides have to agree on which images are held back.
        split = concept.get(SPLIT_KEY) or {}
        return f"{concept.get('path', '')}|{split.get('key_seed', concept.get('seed', 0))}"

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        source_index = self.indices[index]

        return {
            self.path_out_name: self._get_previous_item(variation, self.path_in_name, source_index),
            self.concept_out_name: self._get_previous_item(
                variation, self.concept_in_name, source_index
            ),
        }
