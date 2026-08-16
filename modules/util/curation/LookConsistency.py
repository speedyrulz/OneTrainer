"""Finds the images in a dataset that look unlike the rest of it.

These are usually genuine and worth keeping — a tight angle, a profile, heavy occlusion — but they
pull hardest exactly while the identity is still forming. The curation warm-up eases them in at a
reduced learning rate that ramps to full over the first epochs, so they refine the identity instead
of fighting it, and they are released early the moment they start improving on their own.

This is the *prior*: it covers the epochs before the loss watch has a trend to act on. The loss watch
takes over from there.

Scores are computed once with CLIP (already an OneTrainer dependency, used by the aesthetic scorer)
and cached to a ``look_scores.json`` beside the images. A ``fizgig_look_scores.json`` written by
Fizgig's Look Filter is read too, so a dataset already scored there needs no rescoring.
"""

import json
import logging
import os

from modules.util import path_util
from modules.util.clip_util import single_image_embedding

import torch
from torchvision.transforms import transforms

from PIL import Image

logger = logging.getLogger(__name__)

SCORES_FILENAME = "look_scores.json"
FIZGIG_SCORES_FILENAME = "fizgig_look_scores.json"

# How far below the middle of the pack an image has to sit to count as an outlier, in robust
# standard deviations. 1.5 is loose enough to catch the genuinely odd framings without sweeping in
# half a varied dataset.
DEFAULT_CUTOFF_DEVIATIONS = 1.5

# scale factor that turns a median absolute deviation into a standard-deviation equivalent
_MAD_TO_SIGMA = 1.4826


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _read_scores_file(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
    except Exception:
        logger.warning("[look] could not read %s — ignoring", path)
        return None
    if isinstance(data, dict) and isinstance(data.get("scores"), dict):
        return data
    return None


def load_cached_scores(image_paths: list[str]) -> dict[str, float]:
    """Scores already on disk for these images, from either sidecar format.

    Fizgig stores its keys as bare filenames; ours are full paths. Both are matched by filename so a
    dataset scored in either tool is understood here.
    """
    scores: dict[str, float] = {}
    by_name: dict[str, str] = {}
    for path in image_paths:
        by_name.setdefault(os.path.basename(path), path)
        by_name.setdefault(os.path.splitext(os.path.basename(path))[0], path)

    directories = {os.path.dirname(p) for p in image_paths if os.path.dirname(p)}
    for directory in sorted(directories):
        for filename in (SCORES_FILENAME, FIZGIG_SCORES_FILENAME):
            path = os.path.join(directory, filename)
            if not os.path.exists(path):
                continue
            data = _read_scores_file(path)
            if not data:
                continue
            for key, value in data["scores"].items():
                if not isinstance(value, (int, float)):
                    continue
                resolved = key if key in set(image_paths) else by_name.get(os.path.basename(str(key)))
                if resolved is None:
                    resolved = by_name.get(os.path.splitext(os.path.basename(str(key)))[0])
                if resolved is not None:
                    scores[resolved] = float(value)

    return scores


def save_scores(scores: dict[str, float], cutoff: float) -> None:
    """Write one sidecar per directory so the scoring pass only ever runs once."""
    by_directory: dict[str, dict[str, float]] = {}
    for key, value in scores.items():
        directory = os.path.dirname(key)
        if directory:
            by_directory.setdefault(directory, {})[key] = value

    for directory, entries in by_directory.items():
        path = os.path.join(directory, SCORES_FILENAME)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"cutoff": cutoff, "scores": entries}, f, indent=2)
        except Exception:
            logger.warning("[look] could not write %s", path, exc_info=True)


@torch.no_grad()
def score_images(image_paths: list[str], device: torch.device) -> dict[str, float]:
    """How much each image looks like the rest of the set, as a cosine similarity in [-1, 1].

    Every image is embedded with CLIP and compared against the *median* embedding rather than the
    mean, so a handful of oddities cannot drag the reference point towards themselves and make the
    normal images look like the outliers.
    """
    from transformers import CLIPModel

    if not image_paths:
        return {}

    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    try:
        embeddings: dict[str, torch.Tensor] = {}
        for path in image_paths:
            try:
                with Image.open(path) as image:
                    tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
                features = single_image_embedding(clip, tensor)
                embeddings[path] = features / features.norm().clamp(min=1e-8)
            except Exception:  # noqa: PERF203 - one unreadable image must not stop the scan
                logger.warning("[look] could not embed %s — skipping", os.path.basename(path))

        if len(embeddings) < 3:
            # a reference point drawn from one or two images says nothing about either of them
            return {}

        stacked = torch.stack(list(embeddings.values()))
        reference = stacked.median(dim=0).values
        reference = reference / reference.norm().clamp(min=1e-8)

        return {path: float(torch.dot(vector, reference)) for path, vector in embeddings.items()}
    finally:
        del clip
        if device.type == "cuda":
            torch.cuda.empty_cache()


def cutoff_for(scores: dict[str, float], deviations: float = DEFAULT_CUTOFF_DEVIATIONS) -> float:
    """Where "unlike the rest" starts, from the spread of the scores themselves.

    Uses the median absolute deviation rather than the standard deviation, because the outliers this
    is looking for would otherwise widen the very threshold meant to catch them.
    """
    values = list(scores.values())
    if len(values) < 3:
        return float("-inf")

    middle = _median(values)
    distances = [abs(value - middle) for value in values]

    scale = _median(distances)
    if scale <= 0:
        # More than half the set scored identically, so the middle of the distribution has no
        # spread to measure. That does not mean the tails have none: fall back to the mean distance,
        # which the ties pull towards zero but do not zero out. Still zero and every image really is
        # identical, so nothing can be unlike the rest.
        scale = sum(distances) / len(distances)
    if scale <= 0:
        return float("-inf")

    return middle - deviations * _MAD_TO_SIGMA * scale


def find_outliers(
        image_paths: list[str],
        device: torch.device,
        *,
        deviations: float = DEFAULT_CUTOFF_DEVIATIONS,
        rescore: bool = False,
) -> tuple[set[str], dict[str, float]]:
    """The images that look unlike the rest, scoring them first if nothing has scored them yet."""
    supported = [
        path for path in image_paths
        if path_util.is_supported_image_extension(os.path.splitext(path)[1])
    ]
    if len(supported) < 3:
        return set(), {}

    scores = {} if rescore else load_cached_scores(supported)
    missing = [path for path in supported if path not in scores]

    if missing:
        logger.info("[look] scoring %d image(s) for look consistency", len(missing))
        scores.update(score_images(missing, device))
        if scores:
            save_scores(scores, cutoff_for(scores, deviations))

    if not scores:
        return set(), {}

    cutoff = cutoff_for(scores, deviations)
    outliers = {path for path, score in scores.items() if score < cutoff}

    # Everything cannot be an outlier: if the cutoff would take most of the set, the set is simply
    # varied and there is no core for the odd ones to be odd against.
    if len(outliers) > len(scores) // 2:
        logger.info("[look] %d of %d images scored as outliers — the dataset is too varied for "
                    "this to mean anything; warm-up disabled", len(outliers), len(scores))
        return set(), scores

    return outliers, scores
