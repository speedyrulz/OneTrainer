"""Bridges OneTrainer's training loop to the per-image loss watch.

The watch itself (:mod:`modules.util.curation.loss_watch`, ported from Fizgig) knows nothing about
tensors, batches or diffusion schedules. This is the adapter: it pulls one loss, one timestep and
one image path per sample out of a training step, feeds them in, and turns the verdicts back into a
weight for each sample's loss.

Weighting the loss is how a per-image learning rate is applied. A single optimizer step cannot use a
different learning rate for each image in a batch, but scaling a sample's loss scales exactly its
contribution to the gradient, which comes to the same thing. An excluded image gets a weight of
zero, which is the batched equivalent of skipping its step.
"""

import contextlib
import gc
import json
import os
import traceback
from collections.abc import Iterable

from modules.util.config.TrainConfig import TrainConfig
from modules.util.curation import LookConsistency
from modules.util.curation.loss_watch import PerImageLossWatch
from modules.util.curation.Recaptioner import Recaptioner, invalidate_text_cache
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor

from tqdm import tqdm


def curation_enabled(config: TrainConfig) -> bool:
    return bool(config.curate_dataset or config.curate_per_image_lr)


class DatasetCuration:
    """Per-step and per-epoch hooks the trainer calls when dataset curation is on."""

    def __init__(self, config: TrainConfig, workspace_dir: str, device: torch.device | None = None):
        self.config = config
        self.workspace_dir = workspace_dir
        self.device = device or torch.device(config.train_device)
        self.apply_lr = bool(config.curate_per_image_lr)

        self.watch = PerImageLossWatch(
            workspace_dir,
            apply_lr=self.apply_lr,
            write_jsonl=bool(config.curate_write_log),
            warmup_epochs=max(0, int(config.curate_warmup_epochs or 0)),
            plateau_patience=max(1, int(config.curate_plateau_patience or 1)),
        )

        self._model_ref = None

        self.recaptioner = None
        if config.curate_recaption:
            self.recaptioner = Recaptioner(
                config.curate_recaption_model,
                self.device,
                trigger_word=config.curate_recaption_trigger,
                precision=config.curate_recaption_precision,
                resident_provider=self._resident_qwen,
            )

        self._seen_keys: set[str] = set()
        self._preflight_done = False
        self._plateau_announced = False

        # captions the user edited in the Problem Images window, picked up at the next boundary
        self._edit_queue = os.path.join(workspace_dir, "loss_log", "caption_edits.json")

    # --- the resident captioner ---------------------------------------------------------------

    def attach_model(self, model) -> None:
        """Give curation a look at the model being trained.

        When that model already conditions on Qwen3-VL (Krea 2), recaptioning borrows the resident
        text encoder instead of loading a second Qwen from the Hub — the same trick Fizgig uses,
        and the difference between a ~4.5GB spike at the boundary and none at all.
        """
        self._model_ref = model

    def _resident_qwen(self):
        """A captioner over the training run's own Qwen3-VL, or None when there is no safe one.

        Every check here fails towards None: the fresh quantized load is always available as the
        fallback, so a resident encoder that is the wrong shape, the wrong dtype, or being trained
        is simply not used rather than risked.
        """
        model = self._model_ref
        try:
            text_encoder = getattr(model, "text_encoder", None) if model is not None else None
            if text_encoder is None or type(text_encoder).__name__ != "Qwen3VLModel":
                return None

            if getattr(text_encoder, "visual", None) is None:
                return None  # no vision tower, so it cannot look at an image

            parameters = list(text_encoder.parameters())
            if not parameters:
                return None
            if any(p.requires_grad for p in parameters):
                # the encoder is being trained; captioning through a half-trained LoRA would write
                # captions shaped by the very weights that are still moving
                return None
            if parameters[0].dtype not in (torch.bfloat16, torch.float16, torch.float32):
                return None  # fp8 and quantized towers cannot run the vision path

            current_device = parameters[0].device
            moved = False
            if current_device != self.device:
                if not self._room_to_move(parameters):
                    return None
                self._text_encoder_to(model, text_encoder, self.device)
                moved = True

            from modules.module.QwenVLModel import QwenVLModel

            captioner = QwenVLModel.from_resident(text_encoder, self.device)
            if moved:
                temp_device = torch.device(self.config.temp_device)
                captioner.on_release(
                    lambda: self._text_encoder_to(model, text_encoder, temp_device))
            tqdm.write("[recaption] reusing the training run's own Qwen3-VL text encoder — "
                       "no separate captioner is loaded")
            return captioner
        except Exception:
            traceback.print_exc()
            tqdm.write("[recaption] could not reuse the resident text encoder; falling back to "
                       "loading the captioner separately")
            return None

    def _room_to_move(self, parameters) -> bool:
        """Whether the offloaded text encoder fits on the training device right now."""
        if self.device.type != "cuda":
            return True

        from modules.module.QwenVLModel import _free_vram_gb

        # collected first, or the reading includes freed-but-cached blocks and undercounts
        gc.collect()
        torch.cuda.empty_cache()

        weights_gb = sum(p.numel() * p.element_size() for p in parameters) / 1024 ** 3
        needed = weights_gb + 1.5  # KV cache and vision activations
        free = _free_vram_gb(self.device)
        if free < needed:
            tqdm.write(f"[recaption] the resident text encoder is offloaded and there is not room "
                       f"to bring it back ({free:.1f}GB free, ~{needed:.1f}GB needed) — trying the "
                       f"quantized captioner instead")
            return False
        return True

    # what each captioner family roughly needs free, for deciding whether the denoiser must step
    # aside; the Qwen paths are computed exactly, these are the rest
    _CAPTIONER_NEED_GB = {"BLIP2": 6.0, "BLIP": 2.5, "WD14_VIT_2": 2.0}

    def _captioning_need_gb(self) -> float:
        """How much free VRAM this boundary's captioning will want, for the path it will take."""
        from modules.module.QwenVLModel import REQUIRED_FREE_GB

        if self.config.curate_recaption_model.name != "QWEN3_VL_4B":
            return self._CAPTIONER_NEED_GB.get(self.config.curate_recaption_model.name, 5.0)

        text_encoder = getattr(self._model_ref, "text_encoder", None) if self._model_ref else None
        if text_encoder is not None and type(text_encoder).__name__ == "Qwen3VLModel":
            parameters = list(text_encoder.parameters())
            usable = (parameters
                      and not any(p.requires_grad for p in parameters)
                      and parameters[0].dtype in (torch.bfloat16, torch.float16, torch.float32))
            if usable:
                if parameters[0].device == self.device:
                    return 0.0  # already resident: captioning costs nothing
                return sum(p.numel() * p.element_size() for p in parameters) / 1024 ** 3 + 1.5

        return REQUIRED_FREE_GB[self.config.curate_recaption_precision]

    def _denoiser_mover(self):
        """The model's own method for moving its denoiser, when it has one."""
        for name in ("transformer_to", "unet_to", "prior_to"):
            mover = getattr(self._model_ref, name, None)
            if callable(mover):
                return mover
        return None

    @contextlib.contextmanager
    def _boundary_room(self):
        """Step the denoiser aside when the card is too full to caption next to it.

        At an epoch boundary the denoiser is idle — no step is in flight and no gradients exist —
        so on a card where neither the resident text encoder nor even the quantized captioner fits
        beside it, it waits out the captioning in system RAM and comes straight back. Two PCIe
        copies buys captioning with the run's own full-precision encoder on cards where nothing
        else would fit at all.
        """
        if self.device.type != "cuda":
            yield
            return

        from modules.module.QwenVLModel import _free_vram_gb

        gc.collect()
        torch.cuda.empty_cache()

        try:
            need = self._captioning_need_gb()
            free = _free_vram_gb(self.device)
            mover = self._denoiser_mover() if free < need else None
        except Exception:
            traceback.print_exc()
            mover = None

        if mover is None:
            yield
            return

        temp_device = torch.device(self.config.temp_device)
        tqdm.write(f"[recaption] {free:.1f}GB free but ~{need:.1f}GB needed — the denoiser is idle "
                   f"at an epoch boundary, so it steps aside to {temp_device} while the captions "
                   f"are rewritten")
        moved = False
        try:
            mover(temp_device)
            moved = True
            torch.cuda.empty_cache()
            yield
        finally:
            if moved:
                mover(self.device)
                gc.collect()
                torch.cuda.empty_cache()

    @staticmethod
    def _text_encoder_to(model, text_encoder, device: torch.device) -> None:
        """Move the text encoder through the model's own method when it has one, so a layer
        offload conductor stays consistent with where the weights actually are."""
        mover = getattr(model, "text_encoder_to", None)
        if callable(mover):
            mover(device)
        else:
            text_encoder.to(device)

    # --- per step ---------------------------------------------------------------------------

    @staticmethod
    def image_keys(batch: dict) -> list[str] | None:
        """One stable key per sample. OneTrainer already carries the source path in the batch."""
        keys = batch.get('image_path')
        if keys is None:
            return None
        if isinstance(keys, str):
            return [keys]
        return [str(k) for k in keys]

    @staticmethod
    def normalised_timesteps(data: dict, sample_count: int) -> list[float] | None:
        """Per-sample timesteps mapped onto [0, 1], which is what the watch buckets on.

        Flow matching hands over sigmas already in that range; diffusion models hand over integer
        indices into the noise schedule. Scaling by the largest value the schedule can produce is
        what makes the two comparable, and a run only ever mixes one kind.
        """
        timesteps = data.get('timestep')
        if timesteps is None:
            return None

        values = timesteps.detach().to(dtype=torch.float32).flatten().tolist()
        if len(values) == 1 and sample_count > 1:
            values = values * sample_count
        if len(values) < sample_count:
            return None

        # Well above any sigma, so a value that merely overshoots 1 slightly is clamped rather than
        # mistaken for a schedule index and divided into nothing.
        if max(values) > 1.5:
            scale = float(data.get('num_train_timesteps') or 1000.0)
            values = [value / scale for value in values]

        return [min(max(value, 0.0), 1.0) for value in values[:sample_count]]

    def sample_weights(self, keys: list[str], device: torch.device, dtype: torch.dtype) -> Tensor:
        """The multiplier for each sample's loss: the per-image learning rate, and 0 to exclude."""
        weights = [
            0.0 if self.watch.is_excluded(key) else float(self.watch.multiplier(key))
            for key in keys
        ]
        return torch.tensor(weights, device=device, dtype=dtype)

    def observe_step(
            self,
            train_progress: TrainProgress,
            keys: list[str],
            sample_losses: Tensor,
            data: dict,
    ) -> None:
        """Record every sample of this step against the image it came from.

        The losses recorded are the raw ones, before any curation weight: the weight is the action
        taken, not the measurement, and feeding back weighted losses would let a throttled image
        look like it had improved.
        """
        timesteps = self.normalised_timesteps(data, len(keys))
        if timesteps is None:
            return

        losses = sample_losses.detach().to(dtype=torch.float32).flatten().tolist()
        if len(losses) < len(keys):
            return

        for index, key in enumerate(keys):
            if self.watch.is_excluded(key):
                continue
            self._seen_keys.add(key)
            self.watch.observe(
                epoch=train_progress.epoch + 1,
                step=train_progress.global_step,
                item_keys=key,
                timestep=timesteps[index],
                loss=losses[index],
            )

    # --- per epoch --------------------------------------------------------------------------

    def epoch_boundary(self, train_progress: TrainProgress, tensorboard=None) -> dict:
        """Adjudicate every image, then report what changed."""
        if not self._preflight_done and self._seen_keys:
            # only now is the real training set known, so this is where exclusions written by an
            # earlier run over the same images come back, and where the look prior is applied
            self.watch.preflight(self._seen_keys)
            self._start_outlier_warmup()
            self._preflight_done = True

        epoch = train_progress.epoch
        verdicts = self.watch.epoch_boundary(epoch) or {}

        # A caption change resets an image's history, so both of these have to happen after the
        # verdicts are in: the edit is the answer to the verdict.
        changed = self._apply_caption_edits()
        changed |= self._auto_recaption(verdicts)
        if changed:
            invalidate_text_cache(self.config.cache_dir)

        self._report(epoch, verdicts, train_progress, tensorboard)
        return verdicts

    # --- look-outlier warm-up ---------------------------------------------------------------

    def _start_outlier_warmup(self) -> None:
        """Ease images that look unlike the rest of the set in, over the first few epochs."""
        if not self.config.curate_warmup_outliers:
            return

        try:
            outliers, _scores = LookConsistency.find_outliers(
                sorted(self._seen_keys),
                self.device,
                deviations=float(self.config.curate_outlier_deviations or 1.5),
            )
        except Exception:
            traceback.print_exc()
            tqdm.write("[curation] look scoring failed; continuing without outlier warm-up")
            return

        if not outliers:
            return

        self.watch.set_warmup_keys(
            sorted(outliers),
            start=float(self.config.curate_outlier_start or 0.4),
            ramp_epochs=max(1, int(self.config.curate_outlier_ramp_epochs or 4)),
        )
        tqdm.write(f"[curation] {len(outliers)} image(s) look unlike the rest and will ease in at "
                   f"x{self.config.curate_outlier_start:g}, ramping to full over "
                   f"{self.config.curate_outlier_ramp_epochs} epoch(s):")
        for key in sorted(outliers)[:8]:
            tqdm.write(f"[curation]   {os.path.basename(key)}")
        if len(outliers) > 8:
            tqdm.write(f"[curation]   ... and {len(outliers) - 8} more")

    # --- caption rewriting ------------------------------------------------------------------

    def _auto_recaption(self, verdicts: dict) -> bool:
        """Rewrite the captions of images the watch has confirmed stuck. Returns whether any changed."""
        if self.recaptioner is None:
            return False

        candidates = [
            key for key, verdict in verdicts.items()
            if verdict == "stuck" and self.recaptioner.attempts_left(key)
        ]
        if not candidates:
            return False

        tqdm.write(f"[curation] rewriting the captions of {len(candidates)} stuck image(s)...")
        with self._boundary_room():
            written = self.recaptioner.recaption(sorted(candidates))

        for key, caption in written.items():
            # a fresh caption earns a fresh start: the old trajectory was about the old caption
            self.watch.reset_key(key)
            tqdm.write(f"[curation]   {os.path.basename(key)}: "
                       f"{caption[:90]}{'…' if len(caption) > 90 else ''}")
            if self.recaptioner.spent(key):
                # both attempts used; if it confirms stuck again it is excluded rather than
                # throttled down the escalation ladder forever
                self.watch.mark_incorrigible(key)

        return bool(written)

    def _apply_caption_edits(self) -> bool:
        """Pick up captions the user edited in the Problem Images window while the run continued.

        The window cannot reach into the training thread, so it leaves the edited keys in a small
        file. A human edit always outranks the machine's verdict, so it also restores the image's
        recaption attempts.
        """
        if not os.path.exists(self._edit_queue):
            return False

        try:
            with open(self._edit_queue, encoding="utf-8") as f:
                keys = json.load(f)
        except Exception:
            keys = []
        finally:
            with contextlib.suppress(OSError):
                os.remove(self._edit_queue)

        applied = False
        for key in keys if isinstance(keys, list) else []:
            key = str(key)
            self.watch.reset_key(key)
            if self.recaptioner is not None:
                self.recaptioner.attempts.pop(key, None)
            tqdm.write(f"[curation] caption edited by hand, {os.path.basename(key)} starts over")
            applied = True

        return applied

    def _report(self, epoch: int, verdicts: dict, train_progress: TrainProgress, tensorboard) -> None:
        counts: dict[str, int] = {}
        for verdict in verdicts.values():
            counts[verdict] = counts.get(verdict, 0) + 1

        if counts:
            summary = ", ".join(f"{count} {verdict}" for verdict, count in sorted(counts.items()))
            tqdm.write(f"[curation] epoch {epoch}: {summary}")

        # the two verdicts worth a name in the console: confirmed problems and the early suspicions
        flagged = sorted(
            key for key, verdict in verdicts.items() if verdict in ("stuck", "suspect")
        )
        for key in flagged[:10]:
            multiplier = self.watch.multiplier(key)
            throttled = f" — training at x{multiplier:.2f}" if self.apply_lr and multiplier < 1.0 else ""
            tqdm.write(f"[curation]   {verdicts[key]}: {os.path.basename(key)}{throttled}")
        if len(flagged) > 10:
            tqdm.write(f"[curation]   ... and {len(flagged) - 10} more")

        excluded = self.excluded_keys()
        if excluded:
            tqdm.write(f"[curation] {len(excluded)} image(s) excluded from training")

        if tensorboard is not None:
            step = train_progress.global_step
            for verdict, count in counts.items():
                tensorboard.add_scalar(f"curation/{verdict.lower()}", count, step)
            tensorboard.add_scalar("curation/excluded", len(excluded), step)

        if self.watch.plateaued and not getattr(self, "_plateau_announced", False):
            self._plateau_announced = True
            best = self.watch.best_epoch_estimate
            certainty = "provisional" if self.watch.plateau_pending else "confirmed"
            tqdm.write("")
            tqdm.write("=" * 78)
            tqdm.write(f"[curation] no image is still improving ({certainty} plateau).")
            if best is not None:
                tqdm.write(f"[curation] learning looks finished around epoch {best}; "
                           f"checkpoints from about there are worth comparing.")
            tqdm.write("=" * 78)
            tqdm.write("")

    # --- reporting --------------------------------------------------------------------------

    def excluded_keys(self) -> set[str]:
        return set(self.watch._excluded)

    def is_excluded(self, key: str) -> bool:
        return self.watch.is_excluded(key)

    def set_warmup_keys(self, keys: Iterable[str], *, start: float = 0.4, ramp_epochs: int = 4) -> None:
        """Ease a set of known-unusual images in, rather than letting them fight the early run."""
        self.watch.set_warmup_keys(list(keys), start=start, ramp_epochs=ramp_epochs)

    def close(self) -> None:
        self.watch.close()
