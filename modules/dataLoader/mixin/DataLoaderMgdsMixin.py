import copy
import json
from abc import ABCMeta

from modules.dataLoader.pipelineModules.ValidationSplit import SPLIT_KEY
from modules.util.config.ConceptConfig import ConceptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ConceptType import ConceptType
from modules.util.TrainProgress import TrainProgress

from mgds.MGDS import MGDS
from mgds.PipelineModule import PipelineState

import torch

# Added to the seed of a concept's validation copy. The cached data is grouped by concept path and
# seed, so without this the training and validation halves of one concept would share a cache group
# and overwrite each other's entries.
VALIDATION_SEED_OFFSET = 1_000_003


def validation_split_enabled(config: TrainConfig) -> bool:
    return config.validation and (config.validation_split_percent or 0) > 0


def select_concepts(config: TrainConfig, concepts: list[dict], is_validation: bool) -> list[dict]:
    """The concepts one side of the pipeline should see.

    Without an automatic split this is just the concepts whose type matches the side. With one, every
    trainable concept also appears on the validation side as a copy, tagged so
    :class:`~modules.dataLoader.pipelineModules.ValidationSplit.ValidationSplit` knows to keep the
    held-back images from it rather than the rest.
    """
    validation_concepts = [c for c in concepts if ConceptType(c['type']) == ConceptType.VALIDATION]
    trainable_concepts = [c for c in concepts if ConceptType(c['type']) != ConceptType.VALIDATION]

    if not validation_split_enabled(config):
        return validation_concepts if is_validation else trainable_concepts

    if not is_validation:
        return [_tag_for_split(c) for c in trainable_concepts]

    return validation_concepts + [_as_validation_copy(c) for c in trainable_concepts]


def _tag_for_split(concept: dict) -> dict:
    tagged = copy.deepcopy(concept)
    tagged[SPLIT_KEY] = {'key_seed': concept.get('seed', 0)}
    return tagged


# augmentations that would make the same validation image look different from one pass to the next.
# A validation loss is only comparable over time if the data behind it does not move.
RANDOM_AUGMENTATION_KEYS = (
    'enable_crop_jitter',
    'enable_random_flip',
    'enable_random_rotate',
    'enable_random_brightness',
    'enable_random_contrast',
    'enable_random_saturation',
    'enable_random_hue',
    'enable_random_circular_mask_shrink',
    'enable_random_mask_rotate_crop',
)


def _as_validation_copy(concept: dict) -> dict:
    copied = _tag_for_split(concept)
    copied['type'] = str(ConceptType.VALIDATION)
    # a distinct seed keeps this copy's cached data separate from the training half's
    copied['seed'] = concept.get('seed', 0) + VALIDATION_SEED_OFFSET

    image = copied.get('image')
    if isinstance(image, dict):
        for key in RANDOM_AUGMENTATION_KEYS:
            if key in image:
                image[key] = False

    text = copied.get('text')
    if isinstance(text, dict):
        for key in ('enable_tag_shuffling', 'caps_randomize_enable'):
            if key in text:
                text[key] = False

    return copied


class DataLoaderMgdsMixin(metaclass=ABCMeta):

    def _create_mgds(
            self,
            config: TrainConfig,
            definition: list,
            train_progress: TrainProgress,
            is_validation: bool = False,
    ):
        concepts = config.concepts
        if concepts is None:
            with open(config.concept_file_name, 'r') as f:
                concepts = [ConceptConfig.default_values().from_dict(c) for c in json.load(f)]

        # convert before passing to MGDS
        concepts = [c.to_dict() for c in concepts]

        concepts = select_concepts(config, concepts, is_validation)

        settings = {
            "target_resolution": config.resolution,
            "target_frames": config.frames,
        }

        # Just defaults for now.
        ds = MGDS(
            torch.device(config.train_device),
            concepts,
            settings,
            definition,
            batch_size=config.batch_size, #local batch size
            state=PipelineState(config.dataloader_threads),
            initial_epoch=train_progress.epoch,
            initial_epoch_sample=train_progress.epoch_sample,
        )

        return ds
