"""Renders what the Repair Studio's sliders currently mean, on a live Krea 2 model.

The engine owns the heavy things: the base model (loaded through OneTrainer's own loader, from the
same config the model tab shows), the sampler, and the hook patchers that apply the opened LoRA at
the sliders' live strengths. The tab owns the thread; every public method here is safe to call from
it, serialised by one lock, and nothing here touches a widget.

Two images come out: the *baseline* (every block at its trained strength — what the LoRA is) and the
*edited* render (what the sliders say). The baseline is cached per settings, so dragging sliders
only ever costs the one render that changed.

Krea 2 first because it is the family this fork trains hardest and its DiT is all-Linear, which the
hook patcher requires. The `supports()` gate is where the next family gets added.
"""

import logging
import os
import threading

from modules.util.repair.lora_file import LoraFile
from modules.util.repair.preview_patcher import PreviewPatcher
from modules.util.repair.SliderState import SliderState
from modules.util.torch_util import torch_gc

import torch

logger = logging.getLogger(__name__)


class PreviewSettings:
    """What one render depends on, besides the sliders."""

    def __init__(self, prompt: str = "", seed: int = 42, width: int = 512, height: int = 512,
                 steps: int = 20, cfg_scale: float | None = None):
        self.prompt = prompt
        self.seed = seed
        self.width = width
        self.height = height
        self.steps = steps
        self.cfg_scale = cfg_scale

    def key(self) -> tuple:
        return (self.prompt, self.seed, self.width, self.height, self.steps, self.cfg_scale)


class Krea2PreviewEngine:
    """Loads the training config's Krea 2 model once and renders slider states against it."""

    def __init__(self, train_config):
        self.train_config = train_config
        self.model = None
        self.sampler = None
        self.primary_patcher = PreviewPatcher("primary")
        self.donor_patcher = PreviewPatcher("donor")
        self._lock = threading.Lock()
        self._baseline_key = None
        self._baseline_image = None

    # --- what this engine can do ------------------------------------------------------------------

    @staticmethod
    def supports(train_config) -> str | None:
        """None when the preview can run for this config, else the reason it cannot."""
        if not train_config.model_type.is_krea2():
            return ("The live preview currently supports Krea 2 only — set the model tab to a "
                    "Krea 2 model, or bake and check the file in your sampler of choice.")
        if not train_config.base_model_name:
            return "Set a base model on the model tab first — the preview renders through it."
        return None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    # --- lifecycle --------------------------------------------------------------------------------

    def load(self) -> str | None:
        """Load the base model and sampler. Returns None, or the reason it could not."""
        with self._lock:
            if self.model is not None:
                return None
            reason = self.supports(self.train_config)
            if reason:
                return reason

            from modules.util import create
            from modules.util.compile_util import init_compile
            from modules.util.enum.TrainingMethod import TrainingMethod

            # dynamo config overrides are thread-local, and this runs on whatever worker thread the
            # view spawned — same fix upstream applied to the sample window (#1687)
            init_compile()

            # the same load the sampling tool does, minus everything training-only — and always the
            # plain base model, never the config's LoRA wrapping: the LoRA being edited is applied
            # by hooks, at the sliders' strengths, not baked into the weights
            config = type(self.train_config).default_values().from_dict(self.train_config.to_dict())
            config.training_method = TrainingMethod.FINE_TUNE
            config.optimizer.optimizer = None
            from modules.util.enum.EMAMode import EMAMode
            config.ema = EMAMode.OFF
            config.continue_last_backup = False

            if config.quantization.cache_dir is None:
                config.quantization.cache_dir = config.cache_dir + "/quantization"
                os.makedirs(config.quantization.cache_dir, exist_ok=True)

            try:
                model_loader = create.create_model_loader(
                    model_type=config.model_type, training_method=config.training_method)
                model_setup = create.create_model_setup(
                    model_type=config.model_type,
                    train_device=torch.device(config.train_device),
                    temp_device=torch.device(config.temp_device),
                    training_method=config.training_method,
                )
                model = model_loader.load(
                    model_type=config.model_type,
                    model_names=config.model_names(),
                    weight_dtypes=config.weight_dtypes(),
                    quantization=config.quantization,
                )
                model.train_config = config
                model_setup.setup_optimizations(model, config)
                model_setup.setup_train_device(model, config)
                model_setup.setup_model(model, config)
                model.eval()

                self.sampler = create.create_model_sampler(
                    train_device=torch.device(config.train_device),
                    temp_device=torch.device(config.temp_device),
                    model=model,
                    model_type=config.model_type,
                    training_method=config.training_method,
                )
                self.model = model
            except Exception as e:
                logger.warning("[preview] could not load the model", exc_info=True)
                self.unload_locked()
                return f"Could not load the preview model: {e}"
            return None

    def unload(self) -> None:
        with self._lock:
            self.unload_locked()

    def unload_locked(self) -> None:
        self.primary_patcher.detach()
        self.donor_patcher.detach()
        if self.model is not None:
            # evict() is BaseModel's conductor-aware move to the temp device; dropping the
            # references afterwards is what actually frees the memory
            evict = getattr(self.model, "evict", None)
            if callable(evict):
                try:
                    evict()
                except Exception:
                    logger.warning("[preview] evicting the model failed", exc_info=True)
        self.model = None
        self.sampler = None
        self._baseline_key = None
        self._baseline_image = None
        torch_gc()

    # --- the LoRA under edit ----------------------------------------------------------------------

    def attach(self, primary: LoraFile | None, donor: LoraFile | None,
               state: SliderState) -> str | None:
        """Hook the opened files onto the model. Returns None, or the reason it could not."""
        with self._lock:
            if self.model is None:
                return "Load the preview model first."
            try:
                self.primary_patcher.detach()
                self.donor_patcher.detach()
                if primary is not None:
                    self.primary_patcher.attach(self.model.transformer, primary, state)
                if donor is not None:
                    self.donor_patcher.attach(self.model.transformer, donor, state)
            except Exception as e:
                logger.warning("[preview] could not attach the LoRA", exc_info=True)
                return str(e)
            # a different LoRA means the old baseline shows the wrong file
            self._baseline_key = None
            self._baseline_image = None
            return None

    def match_summary(self) -> str:
        parts = []
        for label, patcher in (("primary", self.primary_patcher), ("donor", self.donor_patcher)):
            if patcher.attached or patcher.unmatched:
                total = patcher.matched + len(patcher.unmatched)
                parts.append(f"{label}: {patcher.matched}/{total} layers hooked")
        return " · ".join(parts)

    # --- rendering --------------------------------------------------------------------------------

    def render(self, state: SliderState, settings: PreviewSettings):
        """One edited render at the sliders' current strengths. Returns a PIL image."""
        with self._lock:
            self.primary_patcher.set_state(state)
            self.donor_patcher.set_state(state)
            return self._render_locked(settings)

    def render_baseline(self, state: SliderState, settings: PreviewSettings):
        """The LoRA as trained: every block at 1.0. Cached until the settings or the file change."""
        with self._lock:
            if self._baseline_key == settings.key() and self._baseline_image is not None:
                return self._baseline_image

            neutral = SliderState.for_blocks(state.blocks)
            self.primary_patcher.set_state(neutral)
            self.donor_patcher.set_state(neutral)
            try:
                image = self._render_locked(settings)
            finally:
                self.primary_patcher.set_state(state)
                self.donor_patcher.set_state(state)

            self._baseline_key = settings.key()
            self._baseline_image = image
            return image

    def _render_locked(self, settings: PreviewSettings):
        if self.sampler is None:
            raise RuntimeError("the preview model is not loaded")

        from modules.util.compile_util import init_compile

        # every render may arrive on a fresh worker thread, and the dynamo overrides are
        # thread-local — without this a compiled model hits the recompile limit mid-preview
        init_compile()

        from modules.util.config.SampleConfig import SampleConfig
        from modules.util.enum.ImageFormat import ImageFormat

        sample = SampleConfig.default_values(self.train_config.model_type)
        sample.prompt = settings.prompt
        sample.seed = settings.seed
        sample.random_seed = False
        sample.width = settings.width
        sample.height = settings.height
        sample.diffusion_steps = settings.steps
        if settings.cfg_scale is not None:
            sample.cfg_scale = settings.cfg_scale

        destination = os.path.join(
            self.train_config.workspace_dir or ".", "repair_preview", "preview")

        result: dict = {}
        self.sampler.sample(
            sample_config=sample,
            destination=destination,
            image_format=ImageFormat.JPG,
            on_sample=lambda output: result.setdefault("image", output.data),
        )

        image = result.get("image")
        if image is None:
            raise RuntimeError("the sampler produced no image")
        return image
