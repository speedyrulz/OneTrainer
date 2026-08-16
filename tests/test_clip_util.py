"""Tests for getting an image embedding out of CLIP.

This exists because the same one-line mistake shipped twice. `get_image_features` returns a wrapper
object from transformers 5, not a tensor, and the failure is quiet: a ModelOutput is an OrderedDict,
so `[0]` keeps working and hands back the per-patch grid instead of the embedding — right up until
something downstream cares about the rank.

Both callers now go through one helper, and it is tested against both shapes so a version bump either
way is caught here rather than in a training run.

Run with:  pytest tests/test_clip_util.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

torch = pytest.importorskip("torch")

from modules.util.clip_util import image_features, single_image_embedding  # noqa: E402

PROJECTION_DIM = 768
HIDDEN_DIM = 1024
PATCHES = 257


class ModelOutputLike(dict):
    """What transformers 5 returns: an OrderedDict subclass, so [0] silently works."""

    def __init__(self, pooler_output, last_hidden_state):
        super().__init__(last_hidden_state=last_hidden_state, pooler_output=pooler_output)
        self.pooler_output = pooler_output
        self.last_hidden_state = last_hidden_state

    def __getitem__(self, key):
        if key == 0:
            return self.last_hidden_state
        return super().__getitem__(key)


class StubClip:
    def __init__(self, mode: str, embedding=None):
        self.mode = mode
        self.embedding = embedding
        self.seen_kwargs = None

    def get_image_features(self, pixel_values=None, **kwargs):
        self.seen_kwargs = kwargs
        batch = pixel_values.shape[0]
        # derived from the input, not conjured, so the autograd graph connects the way it does
        # through real CLIP; the +1 keeps it away from an all-zero vector whose norm is 0
        pooled = (self.embedding if self.embedding is not None
                  else pixel_values.flatten(1)[:, :PROJECTION_DIM] + 1.0)
        if self.mode == "tensor":  # transformers 4
            return pooled
        return ModelOutputLike(pooled, torch.zeros(batch, PATCHES, HIDDEN_DIM))


def pixels(batch: int = 1):
    return torch.zeros(batch, 3, 224, 224)


# --- the helper -----------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["tensor", "model_output"])
def test_the_projected_embedding_comes_back_whichever_version_this_is(mode):
    features = image_features(StubClip(mode), pixels(2))

    assert features.shape == (2, PROJECTION_DIM)


def test_the_patch_grid_is_not_mistaken_for_the_embedding():
    # this is the bug: [0] on a ModelOutput yields last_hidden_state, which is a plausible-looking
    # tensor of entirely the wrong thing
    features = image_features(StubClip("model_output"), pixels(1))

    assert features.shape[-1] == PROJECTION_DIM
    assert features.ndim == 2
    assert features.shape[1] != HIDDEN_DIM


def test_the_batch_dimension_is_kept():
    # the aesthetic scorer scores a batch at a time and squeezes to one score per image; collapsing
    # the batch here would give every image in a step the first image's score
    features = image_features(StubClip("model_output"), pixels(4))

    assert features.shape[0] == 4


def test_extra_arguments_reach_the_model():
    clip = StubClip("model_output")

    image_features(clip, pixels(1), interpolate_pos_encoding=True)

    assert clip.seen_kwargs == {"interpolate_pos_encoding": True}


# --- the single-image form ------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["tensor", "model_output"])
def test_one_image_comes_back_as_a_vector(mode):
    embedding = single_image_embedding(StubClip(mode), pixels(1))

    assert embedding.ndim == 1
    assert embedding.shape == (PROJECTION_DIM,)


def test_the_vector_is_the_embedding_itself():
    wanted = torch.tensor([[0.5, -0.25, 2.0, 1.0]])

    embedding = single_image_embedding(StubClip("model_output", wanted), pixels(1))

    assert torch.equal(embedding, wanted[0])


def test_a_half_precision_embedding_is_promoted():
    # the cosine similarities downstream are compared against a MAD-derived cutoff, and fp16 noise
    # at that scale is the same order as the differences being measured
    wanted = torch.tensor([[0.5, 0.25]], dtype=torch.float16)

    embedding = single_image_embedding(StubClip("model_output", wanted), pixels(1))

    assert embedding.dtype == torch.float32


def test_an_unexpected_shape_stops_rather_than_guessing():
    class Moved:
        @staticmethod
        def get_image_features(pixel_values=None, **kwargs):
            return torch.zeros(1, PATCHES, HIDDEN_DIM)

    # reshaping this into something 1D would produce numbers that look like scores and mean nothing
    with pytest.raises(RuntimeError, match="clip_util"):
        single_image_embedding(Moved(), pixels(1))


# --- the aesthetic scorer, driven for real ----------------------------------------------------------


def make_scorer(clip):
    """An AestheticScoreModel without the CLIP weights or the pooch download."""
    pytest.importorskip("pooch")  # imported at module scope by AestheticScoreModel

    from modules.module.AestheticScoreModel import AestheticScoreModel, MLPModel

    from torchvision.transforms import transforms

    scorer = AestheticScoreModel.__new__(AestheticScoreModel)
    torch.nn.Module.__init__(scorer)
    scorer.clip = clip
    scorer.mlp_model = MLPModel()
    scorer.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                            std=[0.26862954, 0.26130258, 0.27577711])
    scorer.resize = transforms.Resize(224)
    scorer.crop = transforms.CenterCrop(224)
    scorer.score_target = 10.0
    return scorer


def test_the_aesthetic_scorer_runs_on_a_transformers_5_clip():
    scorer = make_scorer(StubClip("model_output"))

    scores = scorer(torch.zeros(3, 3, 256, 256))

    # before the fix this raised TypeError: linalg_vector_norm() argument 'input' must be Tensor
    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()


def test_the_aesthetic_scorer_still_runs_on_an_older_clip():
    scorer = make_scorer(StubClip("tensor"))

    assert scorer(torch.zeros(2, 3, 256, 256)).shape == (2,)


def test_the_aesthetic_scorer_scores_each_image_in_the_batch_separately():
    # two images whose embeddings differ have to come out with different scores; taking the wrong
    # tensor out of the wrapper would have given the whole batch one value
    embedding = torch.zeros(2, PROJECTION_DIM)
    embedding[0, 0] = 1.0
    embedding[1, 1] = 1.0
    scorer = make_scorer(StubClip("model_output", embedding))

    scores = scorer(torch.zeros(2, 3, 256, 256))

    assert scores[0] != scores[1]


def test_the_aesthetic_score_stays_differentiable():
    # it is a loss weight, so it has to carry a gradient back to the image
    scorer = make_scorer(StubClip("model_output"))
    images = torch.zeros(1, 3, 256, 256, requires_grad=True)

    scorer(images).sum().backward()

    assert images.grad is not None
