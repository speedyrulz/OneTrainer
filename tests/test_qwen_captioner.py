"""Tests for the Qwen3-VL captioner.

Nothing here loads the real 4B model — it is an 8GB download. What is worth testing without it is
everything around the forward pass: that an instructed captioner is given an instruction rather than
a sentence to continue, that a second look at an image asks for more, that a caption generated in the
middle of a training run does not move the training RNG, and that oversized images are cut down
before they reach the vision tower.

Run with:  pytest tests/test_qwen_captioner.py
"""

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from modules.module.QwenVLModel import MEGAPIXELS, QwenVLModel, isolated_rng  # noqa: E402
from modules.util.curation.Recaptioner import (  # noqa: E402
    DETAILED_INITIAL_CAPTION,
    INITIAL_CAPTION,
    MAX_ATTEMPTS,
    Recaptioner,
)
from modules.util.enum.CurationCaptionModel import CurationCaptionModel  # noqa: E402
from modules.util.enum.GenerateCaptionsModel import GenerateCaptionsModel  # noqa: E402

from PIL import Image  # noqa: E402

# --- the RNG guard ---------------------------------------------------------------------------------


def test_captioning_leaves_the_training_noise_stream_where_it_found_it():
    torch.manual_seed(42)
    expected = torch.randn(4)

    torch.manual_seed(42)
    with isolated_rng():
        torch.randn(100)  # stand-in for sampled decoding
    actual = torch.randn(4)

    # otherwise a run that recaptioned would diverge from one that did not, and the same seed would
    # stop reproducing the same model
    assert torch.equal(expected, actual)


def test_the_guard_restores_even_when_generation_fails():
    torch.manual_seed(7)
    expected = torch.randn(2)

    torch.manual_seed(7)
    with pytest.raises(RuntimeError), isolated_rng():
        torch.randn(50)
        raise RuntimeError("out of memory")

    assert torch.equal(expected, torch.randn(2))


def test_each_pass_samples_differently():
    draws = []
    for _ in range(4):
        torch.manual_seed(1234)  # identical starting state every time
        with isolated_rng():
            draws.append(torch.randn(3).tolist())

    # a fresh seed inside the guard is what makes a second look produce a fresh phrasing rather than
    # the identical caption
    assert len({tuple(d) for d in draws}) > 1


# --- image sizing ----------------------------------------------------------------------------------


def test_an_oversized_image_is_cut_down_before_the_vision_tower():
    fitted = QwenVLModel._fit(Image.new("RGB", (4096, 2048)))

    width, height = fitted.size
    assert width * height <= MEGAPIXELS * 1024 * 1024
    assert width / height == pytest.approx(2.0, abs=0.01)  # aspect ratio kept


def test_a_small_image_is_left_exactly_as_it_is():
    original = Image.new("RGB", (512, 512))

    assert QwenVLModel._fit(original) is original


# --- what the captioner is asked for ---------------------------------------------------------------


class FakeQwen:
    """A captioner that carries instructions, the way QwenVLModel does."""

    CAPTION_INSTRUCTION = QwenVLModel.CAPTION_INSTRUCTION
    DETAILED_CAPTION_INSTRUCTION = QwenVLModel.DETAILED_CAPTION_INSTRUCTION

    def __init__(self):
        self.seen: list[str] = []

    def generate_caption(self, sample, initial_caption="", **kwargs):
        self.seen.append(initial_caption)
        return "a woman in a red coat, viewed from behind, face not visible"


class FakeBlip:
    """A continuation captioner, which carries no instructions."""

    def __init__(self):
        self.seen: list[str] = []

    def generate_caption(self, sample, initial_caption="", **kwargs):
        self.seen.append(initial_caption)
        return "a cat"


def make_recaptioner(monkeypatch, model, tmp_path):
    image = tmp_path / "img.png"
    image.write_bytes(b"")
    recaptioner = Recaptioner(CurationCaptionModel.QWEN3_VL_4B, torch.device("cpu"))
    monkeypatch.setattr(recaptioner, "_load_model", lambda: model)
    return recaptioner, str(image)


def test_an_instructed_captioner_is_instructed_not_primed(tmp_path, monkeypatch):
    model = FakeQwen()
    recaptioner, image = make_recaptioner(monkeypatch, model, tmp_path)

    recaptioner.recaption([image])

    assert model.seen == [QwenVLModel.CAPTION_INSTRUCTION]
    assert model.seen[0] != INITIAL_CAPTION


def test_the_second_look_asks_for_everything_visible(tmp_path, monkeypatch):
    model = FakeQwen()
    recaptioner, image = make_recaptioner(monkeypatch, model, tmp_path)

    for _ in range(MAX_ATTEMPTS):
        recaptioner.recaption([image])

    assert model.seen == [
        QwenVLModel.CAPTION_INSTRUCTION,
        QwenVLModel.DETAILED_CAPTION_INSTRUCTION,
    ]


def test_a_continuation_captioner_still_gets_its_prefixes(tmp_path, monkeypatch):
    model = FakeBlip()
    recaptioner, image = make_recaptioner(monkeypatch, model, tmp_path)

    for _ in range(MAX_ATTEMPTS):
        recaptioner.recaption([image])

    # BLIP finishes the sentence you start, so an instruction would end up inside the caption
    assert model.seen == [INITIAL_CAPTION, DETAILED_INITIAL_CAPTION]


def test_the_trigger_word_still_goes_on_the_end(tmp_path, monkeypatch):
    model = FakeQwen()
    recaptioner, image = make_recaptioner(monkeypatch, model, tmp_path)
    recaptioner.trigger_word = "ohwx"

    written = recaptioner.recaption([image])

    assert written[image].endswith(", ohwx")


# --- the generation call ---------------------------------------------------------------------------


class StubProcessor:
    """Records what it is handed and returns something shaped like real inputs."""

    def __init__(self):
        self.prompts: list[str] = []
        self.images: list = []
        self.messages: list = []

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        self.messages.append(messages)
        text = next(part["text"] for part in messages[0]["content"] if part["type"] == "text")
        return f"<|im_start|>user\n{text}<|im_end|>\n"

    def __call__(self, text=None, images=None, return_tensors=None):
        self.prompts.extend(text or [])
        self.images.extend(images or [])

        class Inputs(dict):
            def to(self, _device):
                return self

        return Inputs(input_ids=torch.zeros(1, 5, dtype=torch.long))

    @staticmethod
    def batch_decode(sequences, skip_special_tokens=True):
        return ["  a woman in a red coat,\n  viewed from behind  "]


class StubModel:
    device = torch.device("cpu")

    def __init__(self):
        self.kwargs: dict = {}

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return torch.zeros(1, 12, dtype=torch.long)


def make_captioner(monkeypatch) -> QwenVLModel:
    """A QwenVLModel with the 8GB download replaced by stubs."""
    captioner = QwenVLModel.__new__(QwenVLModel)
    captioner.device = torch.device("cpu")
    captioner.dtype = torch.float32
    captioner.processor = StubProcessor()
    captioner.model = StubModel()
    return captioner


def make_sample(tmp_path, size=(64, 64)):
    path = tmp_path / "img.png"
    Image.new("RGB", size, (200, 30, 30)).save(path)

    sample = types.SimpleNamespace()
    sample.get_image = lambda: Image.open(path).convert("RGB")
    return sample


def test_the_instruction_reaches_the_model_as_a_chat_message(tmp_path, monkeypatch):
    captioner = make_captioner(monkeypatch)

    captioner.generate_caption(make_sample(tmp_path))

    [messages] = captioner.processor.messages
    kinds = [part["type"] for part in messages[0]["content"]]
    assert kinds == ["image", "text"]
    assert messages[0]["content"][1]["text"] == QwenVLModel.CAPTION_INSTRUCTION


def test_a_typed_instruction_replaces_the_default(tmp_path, monkeypatch):
    captioner = make_captioner(monkeypatch)

    captioner.generate_caption(make_sample(tmp_path), initial_caption="Name the art style.")

    assert captioner.processor.messages[0][0]["content"][1]["text"] == "Name the art style."


def test_whitespace_is_collapsed_into_one_line(tmp_path, monkeypatch):
    captioner = make_captioner(monkeypatch)

    caption = captioner.generate_caption(make_sample(tmp_path))

    # captions are written one per line, so a generated newline would split into two captions
    assert caption == "a woman in a red coat, viewed from behind"


def test_the_prefix_and_postfix_are_applied(tmp_path, monkeypatch):
    captioner = make_captioner(monkeypatch)

    caption = captioner.generate_caption(
        make_sample(tmp_path), caption_prefix="photo of ", caption_postfix=", ohwx")

    assert caption == "photo of a woman in a red coat, viewed from behind, ohwx"


def test_decoding_is_sampled_so_a_retry_reads_differently(tmp_path, monkeypatch):
    captioner = make_captioner(monkeypatch)

    captioner.generate_caption(make_sample(tmp_path))

    assert captioner.model.kwargs["do_sample"] is True
    assert captioner.model.kwargs["temperature"] == 0.5


def test_the_detailed_instruction_is_given_room_to_answer(tmp_path, monkeypatch):
    captioner = make_captioner(monkeypatch)
    sample = make_sample(tmp_path)

    captioner.generate_caption(sample)
    short = captioner.model.kwargs["max_new_tokens"]

    captioner.generate_caption(
        sample, initial_caption=QwenVLModel.DETAILED_CAPTION_INSTRUCTION)
    detailed = captioner.model.kwargs["max_new_tokens"]

    # 2-4 sentences do not fit in the budget for one
    assert detailed > short


def test_generating_a_caption_does_not_move_the_training_rng(tmp_path, monkeypatch):
    captioner = make_captioner(monkeypatch)
    sample = make_sample(tmp_path)

    torch.manual_seed(99)
    expected = torch.randn(4)

    torch.manual_seed(99)
    captioner.generate_caption(sample)
    actual = torch.randn(4)

    assert torch.equal(expected, actual)


def test_only_the_generated_tokens_are_decoded(tmp_path, monkeypatch):
    captioner = make_captioner(monkeypatch)
    decoded = []
    captioner.processor.batch_decode = lambda seq, skip_special_tokens=True: (
        decoded.append(seq.shape[1]), ["a caption"])[1]

    captioner.generate_caption(make_sample(tmp_path))

    # 12 generated - 5 prompt tokens; decoding the prompt back would put the instruction in the file
    assert decoded == [7]


# --- loading -----------------------------------------------------------------------------------


def stub_loading(monkeypatch) -> dict:
    """Replace the 8GB download with something that records how it was asked for."""
    import modules.module.QwenVLModel as module

    calls: dict = {}

    class Loaded(StubModel):
        def eval(self):
            return self

        def to(self, device):
            calls["device"] = device
            return self

    def load_model(name, dtype=None, **kwargs):
        calls["model_name"] = name
        calls["dtype"] = dtype
        return Loaded()

    monkeypatch.setattr(module.AutoProcessor, "from_pretrained",
                        classmethod(lambda cls, name, **kw: StubProcessor()))
    monkeypatch.setattr(module.Qwen3VLForConditionalGeneration, "from_pretrained",
                        classmethod(lambda cls, name, **kw: load_model(name, **kw)))
    return calls


def test_fp16_is_corrected_to_bf16(monkeypatch):
    calls = stub_loading(monkeypatch)

    # callers pass fp16 because that is what the other captioners want; Qwen3-VL overflows in it and
    # returns empty or garbled captions, so it is corrected here rather than at every call site
    captioner = QwenVLModel(torch.device("cpu"), torch.float16)

    assert captioner.dtype == torch.bfloat16
    assert calls["dtype"] == torch.bfloat16


def test_a_dtype_that_is_fine_is_left_alone(monkeypatch):
    calls = stub_loading(monkeypatch)

    QwenVLModel(torch.device("cpu"), torch.float32)

    assert calls["dtype"] == torch.float32


def test_it_loads_the_model_fizgig_uses(monkeypatch):
    calls = stub_loading(monkeypatch)

    QwenVLModel(torch.device("cpu"), torch.bfloat16)

    assert calls["model_name"] == "Qwen/Qwen3-VL-4B-Instruct"
    assert calls["device"] == torch.device("cpu")


# --- it is offered everywhere a captioner can be chosen --------------------------------------------


def test_qwen_is_offered_for_recaptioning():
    assert CurationCaptionModel.QWEN3_VL_4B in list(CurationCaptionModel)


def test_qwen_is_offered_in_the_batch_captioning_tool():
    assert GenerateCaptionsModel.QWEN3_VL_4B in list(GenerateCaptionsModel)


@pytest.mark.parametrize("module", [
    "modules.ui.CtkGenerateCaptionsWindowView",
    "modules.ui.PySide6GenerateCaptionsWindowView",
])
def test_the_captioning_window_lists_qwen(module):
    import importlib
    import inspect

    source = inspect.getsource(importlib.import_module(module))

    assert "Qwen3-VL 4B" in source


def test_the_caption_tool_can_load_it():
    # read rather than import: CaptionUIController pulls in the whole masking stack, and what is
    # being checked is one dropdown value reaching one constructor
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "modules", "ui", "CaptionUIController.py"), encoding="utf-8") as f:
        source = f.read()

    assert 'model == "Qwen3-VL 4B"' in source
    assert "QwenVLModel(default_device" in source


@pytest.mark.parametrize("model_type", list(CurationCaptionModel))
def test_every_offered_captioner_has_a_case_of_its_own(model_type):
    """A missing case falls through to the default and silently loads BLIP2 instead."""
    import inspect

    source = inspect.getsource(Recaptioner._load_model)

    assert f"CurationCaptionModel.{model_type.name}" in source


# --- precision and the VRAM guard -----------------------------------------------------------------
#
# The captioner loads while the training model is still resident, and CUDA on Windows does not fail
# an allocation past the card — it pages into shared system memory and the run appears to freeze.
# So the model quantizes by default, and refuses cleanly when even that will not fit.


import modules.module.QwenVLModel as qwen_module  # noqa: E402
from modules.util.enum.CurationCaptionPrecision import CurationCaptionPrecision  # noqa: E402


def stub_quantized_loading(monkeypatch, *, free_gb=100.0, bitsandbytes=True) -> dict:
    calls: dict = {}

    class Loaded(StubModel):
        def eval(self):
            return self

        def to(self, device):
            calls["moved_to"] = device
            return self

    def load_model(name, dtype=None, **kwargs):
        calls["model_name"] = name
        calls["dtype"] = dtype
        calls.update(kwargs)
        return Loaded()

    monkeypatch.setattr(qwen_module.AutoProcessor, "from_pretrained",
                        classmethod(lambda cls, name, **kw: StubProcessor()))
    monkeypatch.setattr(qwen_module.Qwen3VLForConditionalGeneration, "from_pretrained",
                        classmethod(lambda cls, name, **kw: load_model(name, **kw)))
    monkeypatch.setattr(qwen_module, "_bitsandbytes_available", lambda: bitsandbytes)
    monkeypatch.setattr(qwen_module, "_free_vram_gb", lambda device: free_gb)
    return calls


def test_on_cuda_the_captioner_loads_four_bit_by_default(monkeypatch):
    calls = stub_quantized_loading(monkeypatch)

    QwenVLModel(torch.device("cuda"), torch.float16)

    config = calls["quantization_config"]
    assert config.load_in_4bit
    assert config.bnb_4bit_quant_type == "nf4"
    # a quantized model is placed by device_map; .to() on it raises in bitsandbytes
    assert calls["device_map"] == {"": 0}
    assert "moved_to" not in calls


def test_the_named_cuda_device_is_the_one_used(monkeypatch):
    calls = stub_quantized_loading(monkeypatch)

    QwenVLModel(torch.device("cuda:1"), torch.bfloat16)

    assert calls["device_map"] == {"": 1}


def test_int8_is_available_for_cards_with_room(monkeypatch):
    calls = stub_quantized_loading(monkeypatch)

    QwenVLModel(torch.device("cuda"), torch.bfloat16,
                precision=CurationCaptionPrecision.INT8)

    assert calls["quantization_config"].load_in_8bit


def test_bf16_loads_full_weights_the_old_way(monkeypatch):
    calls = stub_quantized_loading(monkeypatch)

    QwenVLModel(torch.device("cuda"), torch.bfloat16,
                precision=CurationCaptionPrecision.BF16)

    assert "quantization_config" not in calls
    assert calls["moved_to"] == torch.device("cuda")


def test_on_cpu_no_quantization_is_attempted(monkeypatch):
    # bitsandbytes needs CUDA; on CPU the full-weights path is the only one
    calls = stub_quantized_loading(monkeypatch)

    QwenVLModel(torch.device("cpu"), torch.float32)

    assert "quantization_config" not in calls


def test_without_bitsandbytes_it_falls_back_to_full_weights(monkeypatch):
    calls = stub_quantized_loading(monkeypatch, bitsandbytes=False)

    QwenVLModel(torch.device("cuda"), torch.bfloat16)

    assert "quantization_config" not in calls
    assert calls["moved_to"] == torch.device("cuda")


def test_a_card_without_room_refuses_instead_of_paging(monkeypatch):
    stub_quantized_loading(monkeypatch, free_gb=2.0)

    # the alternative on Windows is WDDM quietly spilling into shared memory, which does not error —
    # it makes the whole run crawl, which reads as a freeze
    with pytest.raises(RuntimeError, match="VRAM"):
        QwenVLModel(torch.device("cuda"), torch.bfloat16)


def test_the_refusal_names_the_setting_to_change(monkeypatch):
    stub_quantized_loading(monkeypatch, free_gb=2.0)

    with pytest.raises(RuntimeError, match="Recaption Precision"):
        QwenVLModel(torch.device("cuda"), torch.bfloat16)


def test_the_guard_judges_the_precision_actually_loading(monkeypatch):
    # 5GB free: enough for NF4 (4.5), not for the BF16 fallback (10.5) that no bitsandbytes forces
    stub_quantized_loading(monkeypatch, free_gb=5.0, bitsandbytes=False)

    with pytest.raises(RuntimeError, match="VRAM"):
        QwenVLModel(torch.device("cuda"), torch.bfloat16)


def test_five_gb_free_is_plenty_for_nf4(monkeypatch):
    calls = stub_quantized_loading(monkeypatch, free_gb=5.0)

    QwenVLModel(torch.device("cuda"), torch.bfloat16)

    assert calls["quantization_config"].load_in_4bit


def test_a_failed_load_skips_the_boundary_rather_than_stalling(tmp_path, monkeypatch):
    # the Recaptioner already treats a load failure as "skip this boundary"; the guard's refusal
    # must travel that path, not crash the trainer
    image = str(tmp_path / "img.png")
    open(image, "wb").close()
    stub_quantized_loading(monkeypatch, free_gb=2.0)
    recaptioner = Recaptioner(CurationCaptionModel.QWEN3_VL_4B, torch.device("cuda"))

    assert recaptioner.recaption([image]) == {}
    assert recaptioner.attempts_left(image)


def test_the_configured_precision_reaches_the_captioner(monkeypatch):
    seen = {}

    class FakeQwenVL:
        def __init__(self, device, dtype, precision=None):
            seen["precision"] = precision

    monkeypatch.setattr(qwen_module, "QwenVLModel", FakeQwenVL)
    recaptioner = Recaptioner(CurationCaptionModel.QWEN3_VL_4B, torch.device("cpu"),
                              precision=CurationCaptionPrecision.INT8)

    recaptioner._load_model()

    assert seen["precision"] == CurationCaptionPrecision.INT8


def test_every_precision_has_a_vram_budget():
    from modules.module.QwenVLModel import REQUIRED_FREE_GB

    assert set(REQUIRED_FREE_GB) == set(CurationCaptionPrecision)
