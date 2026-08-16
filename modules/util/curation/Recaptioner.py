"""Rewrites the captions of images that training has decided are stuck.

When the loss watch confirms an image is stuck, the usual cause is that its caption does not describe
what is in the picture. This looks at the image again with one of OneTrainer's captioning models,
writes what is actually visible, and hands the image back to training with a clean slate.

Each image gets two attempts. The second goes for exhaustive detail, on the grounds that the first
caption demonstrably was not enough. An image still stuck after both has had its benefit of the
doubt: the watch excludes it, so the run stops carrying its error, and the exclusion is remembered
for next time.

The trigger word, when set, is appended at the *end* of the caption. A trailing token is a much
weaker identity claim than a leading one, which matters for an image the model is already struggling
to reconcile.
"""

import logging
import os
import shutil

from modules.module.BaseImageCaptionModel import CaptionSample
from modules.util.enum.CurationCaptionModel import CurationCaptionModel
from modules.util.enum.CurationCaptionPrecision import CurationCaptionPrecision

import torch

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 2

# What a *continuation* captioner (BLIP, BLIP2) is asked to start from. The second pass asks for
# detail because the first caption is now known to have failed. An *instructed* captioner such as
# Qwen3-VL carries its own two instructions instead — see `_instruction_for`.
INITIAL_CAPTION = ""
DETAILED_INITIAL_CAPTION = "a detailed description of"


class Recaptioner:
    """Loads a captioning model on demand, rewrites captions, and keeps count of attempts."""

    def __init__(
            self,
            model_type: CurationCaptionModel,
            device: torch.device,
            *,
            trigger_word: str = "",
            precision: CurationCaptionPrecision = CurationCaptionPrecision.NF4,
            resident_provider=None,
    ):
        self.model_type = model_type
        self.device = device
        self.trigger_word = (trigger_word or "").strip()
        self.precision = precision
        # a zero-argument callable returning a ready captioner over the training run's own
        # vision-language model, or None; tried before downloading a second one
        self.resident_provider = resident_provider
        self.attempts: dict[str, int] = {}

    def attempts_left(self, key: str) -> bool:
        return self.attempts.get(key, 0) < MAX_ATTEMPTS

    def _load_model(self):
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        match self.model_type:
            case CurationCaptionModel.BLIP2:
                from modules.module.Blip2Model import Blip2Model
                return Blip2Model(self.device, dtype)
            case CurationCaptionModel.BLIP:
                from modules.module.BlipModel import BlipModel
                return BlipModel(self.device, dtype)
            case CurationCaptionModel.WD14_VIT_2:
                from modules.module.WDModel import WDModel
                return WDModel(self.device, dtype)
            case CurationCaptionModel.QWEN3_VL_4B:
                if self.resident_provider is not None:
                    resident = self.resident_provider()
                    if resident is not None:
                        return resident
                from modules.module.QwenVLModel import QwenVLModel
                return QwenVLModel(self.device, dtype, precision=self.precision)
            case _:
                from modules.module.Blip2Model import Blip2Model
                return Blip2Model(self.device, dtype)

    def recaption(self, keys: list[str]) -> dict[str, str]:
        """Rewrite the captions of these images. Returns {image path: new caption} for what changed.

        The model is loaded for the batch and freed afterwards, because this runs between epochs
        while the training model is still resident and the VRAM is not ours to keep.
        """
        todo = [key for key in keys if self.attempts_left(key) and os.path.exists(key)]
        if not todo:
            return {}

        try:
            model = self._load_model()
        except Exception:
            logger.warning("[recaption] could not load the captioning model — skipping this "
                           "boundary", exc_info=True)
            return {}

        written: dict[str, str] = {}
        try:
            for key in todo:
                attempt = self.attempts.get(key, 0) + 1
                try:
                    caption = self._caption_one(model, key, attempt)
                except Exception:
                    logger.warning("[recaption] captioning failed for %s — will retry next "
                                   "boundary", os.path.basename(key), exc_info=True)
                    continue

                if not caption:
                    continue

                if self._write_caption(key, caption):
                    # counted only once the caption is actually on disk, so a failed write leaves
                    # the attempt available rather than silently burning it
                    self.attempts[key] = attempt
                    written[key] = caption
                    logger.info("[recaption] %s (attempt %d/%d%s): %s",
                                os.path.basename(key), attempt, MAX_ATTEMPTS,
                                ", detailed" if attempt >= MAX_ATTEMPTS else "",
                                caption[:110] + ("…" if len(caption) > 110 else ""))
        finally:
            release = getattr(model, "release", None)
            if release is not None:
                # a resident captioner puts the text encoder back where the training loop keeps
                # it; an owned one just drops its weights
                release()
            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return written

    @staticmethod
    def _instruction_for(model, detailed: bool) -> str:
        """What to send as `initial_caption`, which each captioner reads in its own way.

        BLIP and BLIP2 continue from it, so it has to be the opening words of a caption. Qwen3-VL is
        instructed rather than primed, so it carries a pair of real instructions on the class and
        those are used instead. Anything else falls back to the continuation prompts.
        """
        name = "DETAILED_CAPTION_INSTRUCTION" if detailed else "CAPTION_INSTRUCTION"
        fallback = DETAILED_INITIAL_CAPTION if detailed else INITIAL_CAPTION
        return getattr(model, name, fallback)

    def _caption_one(self, model, key: str, attempt: int) -> str:
        sample = CaptionSample(key)
        initial = self._instruction_for(model, detailed=attempt >= MAX_ATTEMPTS)
        caption = (model.generate_caption(sample, initial_caption=initial) or "").strip()

        if caption and self.trigger_word:
            caption = f"{caption}, {self.trigger_word}"
        return caption

    @staticmethod
    def caption_path(key: str) -> str:
        return os.path.splitext(key)[0] + ".txt"

    def _write_caption(self, key: str, caption: str) -> bool:
        path = self.caption_path(key)
        try:
            # keep the caption training replaced, so a bad rewrite can be undone by hand
            if os.path.exists(path) and not os.path.exists(path + ".orig"):
                shutil.copy2(path, path + ".orig")
            with open(path, "w", encoding="utf-8") as f:
                f.write(caption)
            return True
        except Exception:
            logger.warning("[recaption] could not write %s", path, exc_info=True)
            return False

    def spent(self, key: str) -> bool:
        """True once an image has used both attempts, so the watch should stop forgiving it."""
        return self.attempts.get(key, 0) >= MAX_ATTEMPTS


def invalidate_text_cache(cache_dir: str) -> bool:
    """Drop the cached text embeddings so the rewritten captions are encoded afresh next epoch.

    The whole text cache goes rather than the entries for the changed images: it is keyed by concept
    and position, not by caption, so there is no way to reach into it for one image. Re-encoding text
    is cheap next to the image latents, which are left alone.
    """
    text_cache = os.path.join(cache_dir, "text")
    if not os.path.isdir(text_cache):
        return False
    try:
        shutil.rmtree(text_cache)
        logger.info("[recaption] cleared the text cache; captions will be re-encoded next epoch")
        return True
    except Exception:
        logger.warning("[recaption] could not clear the text cache at %s — the rewritten captions "
                       "will not take effect until it is cleared by hand", text_cache,
                       exc_info=True)
        return False
