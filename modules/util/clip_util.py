"""Getting an image embedding out of CLIP, across transformers versions.

`CLIPModel.get_image_features` used to return the projected image embedding as a plain tensor. From
transformers 5 it returns a `BaseModelOutputWithPooling` and the embedding moved to `pooler_output`.

That change does not fail loudly. A `ModelOutput` subclasses `OrderedDict`, so code that indexes the
result with `[0]` keeps working and quietly gets `last_hidden_state` instead — the per-patch grid,
`[batch, seq, hidden]` rather than `[batch, projection_dim]`. Wrong rank, wrong meaning, and no error
until something further downstream happens to care about the shape. Code that treats the result as a
tensor at all (`embedding / norm(embedding)`) raises a TypeError naming a type nobody was expecting.

One helper, so there is one place to change when it moves again.
"""

from torch import Tensor


def image_features(clip, pixel_values: Tensor, **kwargs) -> Tensor:
    """The projected CLIP image embedding, `[batch, projection_dim]`.

    Accepts either the modern `BaseModelOutputWithPooling` or the bare tensor older versions
    returned, so this works whichever transformers happens to be installed.
    """
    features = clip.get_image_features(pixel_values=pixel_values, **kwargs)

    # getattr rather than isinstance: the output classes get renamed and moved between versions, but
    # the attribute has been stable, and a plain tensor simply does not have it
    return getattr(features, "pooler_output", features)


def single_image_embedding(clip, pixel_values: Tensor, **kwargs) -> Tensor:
    """The embedding for one image, as a 1D vector.

    Raises rather than reshaping if CLIP returns something unexpected: a silently wrong-shaped
    embedding produces plausible-looking numbers that mean nothing, which is worse than a stop.
    """
    features = image_features(clip, pixel_values, **kwargs).float()

    if features.ndim == 2 and features.shape[0] == 1:
        features = features[0]
    if features.ndim != 1:
        raise RuntimeError(
            f"CLIP returned a {features.ndim}D image embedding of shape {tuple(features.shape)} "
            f"where a single vector was expected. The transformers API has moved again; "
            f"modules/util/clip_util.py needs updating.")
    return features
