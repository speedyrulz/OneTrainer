"""Captioning with Qwen3-VL — a vision-language model that is told what to write, not primed.

BLIP and BLIP2 are *continuation* captioners: you hand them the first few words and they finish the
sentence. Qwen3-VL is instructed instead, so `initial_caption` is read here as the instruction rather
than as a prefix, and when it is empty the default instruction below is used.

The instructions and the decoding settings are taken from
[Fizgig](https://github.com/shootthesound/Fizgig) by Peter Neill (Apache 2.0), which uses this same
model to rewrite the captions of images training has decided are stuck. See
[Dataset Curation](../../docs/DatasetCuration.md).
"""

import contextlib
import importlib.util
import logging
import random

from modules.module.BaseImageCaptionModel import BaseImageCaptionModel, CaptionSample
from modules.util.enum.CurationCaptionPrecision import CurationCaptionPrecision

import torch

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from PIL import Image

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"

# How much image the vision tower is given. A training image is often far larger than the model can
# use, and the token count grows with the pixel count for nothing.
MEGAPIXELS = 1.0

# What each precision needs free, measured generously: weights plus KV cache and vision activations.
# The margin matters because running out mid-generation does not error on Windows — WDDM pages into
# shared system memory instead and the GPU crawls, which reads as a frozen training run.
REQUIRED_FREE_GB = {
    CurationCaptionPrecision.NF4: 4.5,
    CurationCaptionPrecision.INT8: 6.5,
    CurationCaptionPrecision.BF16: 10.5,
}


def _bitsandbytes_available() -> bool:
    return importlib.util.find_spec("bitsandbytes") is not None


def build_generation_shell(text_encoder, processor):
    """Give an encoder-only Qwen3VLModel back its voice, without copying a single weight.

    A training run that conditions on Qwen3-VL (Krea 2) loads `Qwen3VLModel` — hidden states only, no
    LM head, no `generate()`. But Qwen3-VL-4B *ties* its LM head to the input embeddings; the
    checkpoint does not even ship `lm_head.weight`. So the full text-generating model is sitting in
    the encoder already, minus a matmul against a matrix it also already has.

    This builds a `Qwen3VLForConditionalGeneration` on the meta device (no allocation), splices the
    resident encoder in as its `.model`, and points `lm_head.weight` at the encoder's embedding
    Parameter. The result runs transformers' real generation loop — sampling, KV cache, Qwen's
    multimodal rope handling — over the training run's own weights, and costs nothing but the shell.

    The tie is done by sharing the Parameter directly. `tie_weights()` is NOT used: called on a
    hand-assembled model it quietly leaves the head untied, which produces confident nonsense
    through a randomly initialised matrix rather than an error.
    """
    from transformers import Qwen3VLForConditionalGeneration

    text_config = text_encoder.config.get_text_config()
    if not getattr(text_config, "tie_word_embeddings", False):
        raise RuntimeError(
            "this Qwen3-VL variant does not tie its LM head to the embeddings, so the resident "
            "encoder cannot generate — a separate captioner has to load")

    # the constructor resolves and writes defaults into the config it is handed, and this one
    # belongs to the model being trained — it gets a copy, not the original
    import copy

    with torch.device("meta"):
        shell = Qwen3VLForConditionalGeneration(copy.deepcopy(text_encoder.config))

    shell.model = text_encoder

    embedding = text_encoder.get_input_embeddings()
    head = torch.nn.Linear(embedding.weight.shape[1], embedding.weight.shape[0],
                           bias=False, device="meta")
    head.weight = embedding.weight
    shell.lm_head = head

    stray = [name for name, p in shell.named_parameters() if p.device.type == "meta"]
    if stray:
        raise RuntimeError(f"the generation shell still has unfilled weights: {stray[:4]} — the "
                           f"transformers layout has changed and this graft needs updating")

    tokenizer = getattr(processor, "tokenizer", None)
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        shell.generation_config.eos_token_id = eos
        if shell.generation_config.pad_token_id is None:
            shell.generation_config.pad_token_id = eos

    return shell.eval()


def _free_vram_gb(device: torch.device) -> float:
    free, _total = torch.cuda.mem_get_info(device)
    return free / 1024 ** 3


def _require_free_vram(device: torch.device, precision: CurationCaptionPrecision) -> None:
    """Refuse to load into VRAM that is not there.

    CUDA over-commits gracefully on Windows: an allocation past the card spills into shared system
    memory and the GPU crawls rather than erroring. For a mid-run captioner that is the worst
    outcome — the training run appears to hang. A clean refusal lets the caller skip the boundary
    and tell the user what to change.
    """
    if device.type != "cuda":
        return

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    needed = REQUIRED_FREE_GB[precision]
    free = _free_vram_gb(device)
    if free < needed:
        raise RuntimeError(
            f"Not enough free VRAM to load the captioner: {free:.1f}GB free, about {needed:.1f}GB "
            f"needed at {precision} precision with the training model still resident. Lower the "
            f"Recaption Precision on the training tab, or reduce the training run's own VRAM use.")


@contextlib.contextmanager
def isolated_rng():
    """Keep sampled decoding out of the training noise stream.

    Captioning can run *during* a training run, at an epoch boundary. Sampling draws from the global
    torch RNG, so without this a run that recaptioned would take a different noise path from one that
    did not, and the same seed would stop reproducing the same model.
    """
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(random.randint(1, 2 ** 31 - 1))
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


class QwenVLModel(BaseImageCaptionModel):
    # What to write when nothing more specific is asked for. Names the viewpoint and whether the face
    # is visible, because those are the two things a caption most often omits and a diffusion model
    # most needs told.
    CAPTION_INSTRUCTION = (
        "Write one factual training caption for this image as a single sentence. Describe the "
        "subject, their pose and clothing, the camera viewpoint (e.g. 'viewed from behind', 'side "
        "profile', 'close-up'), whether the face is visible, and the setting. State only what is "
        "visible — no speculation, no names, no style commentary."
    )

    # For a second look at an image the first caption failed to fix: the miss is probably something
    # salient the short caption skipped, so name every visible element that could contradict the
    # conditioning.
    DETAILED_CAPTION_INSTRUCTION = (
        "Write a detailed factual training caption for this image, 2-4 sentences. Cover: the subject "
        "and exactly how much of them is visible (state the camera viewpoint and explicitly whether "
        "the face is visible or hidden), their pose and body position, every visible clothing item "
        "with colors, hair style and color, any objects they hold or touch, anything partially "
        "blocking or cropping the subject, the lighting, and the background/setting with its main "
        "objects. State only what is visible — no speculation, no names, no style commentary."
    )

    def __init__(
            self,
            device: torch.device,
            dtype: torch.dtype,
            model_name: str = MODEL_NAME,
            precision: "CurationCaptionPrecision | None" = None,
    ):
        self.device = device
        # Qwen3-VL overflows in fp16 and returns empty or garbled captions. Callers pass fp16 because
        # that is what the other captioners want, so it is corrected here rather than at every one.
        self.dtype = torch.bfloat16 if dtype == torch.float16 else dtype

        if precision is None:
            precision = CurationCaptionPrecision.NF4

        self.processor = AutoProcessor.from_pretrained(model_name)

        load_kwargs, quantized, effective = self._load_kwargs(precision)
        _require_free_vram(self.device, effective)

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=self.dtype,
            **load_kwargs,
        )
        self.model.eval()
        if not quantized:
            # a bitsandbytes-quantized model is placed by device_map and refuses .to()
            self.model.to(self.device)

        self._restore = None
        self._was_training = None

    @classmethod
    def from_resident(cls, text_encoder, device: torch.device, processor=None) -> "QwenVLModel":
        """A captioner over a training run's own resident Qwen3-VL, loading no weights at all.

        This is how Fizgig captions during a Krea 2 run: the model that conditions the training
        *is* a vision-language model, so nothing new needs VRAM. Only the processor (a few small
        config files, cached after the first download) comes from the Hub.
        """
        self = cls.__new__(cls)
        self.device = device
        self.dtype = next(text_encoder.parameters()).dtype
        self.processor = processor if processor is not None else AutoProcessor.from_pretrained(MODEL_NAME)

        # recorded before the shell's eval() flips it, restored on release so a text encoder that
        # was in train mode goes back the way it was found
        self._was_training = text_encoder.training
        self._restore = None
        self.model = build_generation_shell(text_encoder, self.processor)
        return self

    def on_release(self, callback) -> None:
        """What to do when the captioner is done — the resident path uses this to move the text
        encoder back where the training loop keeps it."""
        self._restore = callback

    def release(self) -> None:
        model = self.model
        self.model = None
        if self._was_training is not None and model is not None:
            # shell.model is the resident text encoder itself
            model.model.train(self._was_training)
        if self._restore is not None:
            restore, self._restore = self._restore, None
            restore()

    def _load_kwargs(self, precision: "CurationCaptionPrecision") -> tuple[dict, bool, "CurationCaptionPrecision"]:
        """How to load at the asked-for precision, and the precision actually achievable.

        Quantization needs CUDA and bitsandbytes; anywhere that does not hold, the load falls back
        to full weights rather than failing — the VRAM guard then judges *that* footprint.
        """
        if self.device.type != "cuda" or precision == CurationCaptionPrecision.BF16:
            return {}, False, CurationCaptionPrecision.BF16

        if not _bitsandbytes_available():
            logger.warning("[caption] bitsandbytes is not installed, so the captioner cannot be "
                           "quantized — loading full %s weights instead", self.dtype)
            return {}, False, CurationCaptionPrecision.BF16

        from transformers import BitsAndBytesConfig

        if precision == CurationCaptionPrecision.INT8:
            config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.dtype,
            )

        device_index = self.device.index if self.device.index is not None else 0
        return {"quantization_config": config, "device_map": {"": device_index}}, True, precision

    def generate_caption(
            self,
            caption_sample: CaptionSample,
            initial_caption: str = "",
            caption_prefix: str = "",
            caption_postfix: str = "",
    ) -> str:
        instruction = initial_caption.strip() or self.CAPTION_INSTRUCTION
        # the long instruction asks for several sentences, so it needs the room to write them
        max_new_tokens = 240 if instruction == self.DETAILED_CAPTION_INSTRUCTION else 120

        image = self._fit(caption_sample.get_image())

        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction},
        ]}]
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt") \
            .to(self.model.device)

        with torch.no_grad(), isolated_rng():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                # Sampled rather than greedy so a second look at the same image produces a fresh
                # phrasing instead of the identical caption. The temperature is kept low: the
                # variation wanted is in the wording, not in how much the model is willing to claim.
                do_sample=True,
                temperature=0.5,
                top_p=0.9,
            )

        generated = outputs[:, inputs["input_ids"].shape[1]:]
        predicted_caption = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        predicted_caption = " ".join(predicted_caption.split())

        return (caption_prefix + predicted_caption + caption_postfix).strip()

    @staticmethod
    def _fit(image: Image.Image) -> Image.Image:
        """Downscale to at most `MEGAPIXELS`, never upscale."""
        cap = int(MEGAPIXELS * 1024 * 1024)
        width, height = image.size
        if width * height <= cap or width <= 0 or height <= 0:
            return image
        scale = (cap / (width * height)) ** 0.5
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return image.resize(size, Image.Resampling.LANCZOS)
