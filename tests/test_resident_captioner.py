"""Tests for captioning through a training run's own Qwen3-VL (the Krea 2 path).

Krea 2's text encoder is Qwen3-VL — a vision-language model that OneTrainer loads encoder-only, with
no LM head. Because Qwen3-VL-4B ties its head to the input embeddings, a generation shell grafted
around the resident encoder is the complete model, with no weights loaded and no VRAM spent. That
graft is the risky part, so it is tested against a real (tiny) Qwen3-VL and judged by measurement:
the shell's logits must equal hidden-states-times-embeddings exactly.

The pitfall that makes measurement non-negotiable: `tie_weights()` on a hand-assembled model quietly
leaves the head untied, and an untied head produces confident text through a randomly initialised
matrix. Nothing errors; the captions are just nonsense.

Run with:  pytest tests/test_resident_captioner.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from modules.module.QwenVLModel import QwenVLModel, build_generation_shell  # noqa: E402
from modules.util.curation.DatasetCuration import DatasetCuration  # noqa: E402
from modules.util.curation.Recaptioner import Recaptioner  # noqa: E402
from modules.util.enum.CurationCaptionModel import CurationCaptionModel  # noqa: E402
from tests.test_dataset_curation import make_config  # noqa: E402

# --- a tiny real Qwen3-VL -------------------------------------------------------------------------


def tiny_config(tie: bool = True):
    from transformers import Qwen3VLConfig

    return Qwen3VLConfig(
        text_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
                     "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 160,
                     "tie_word_embeddings": tie, "max_position_embeddings": 512, "head_dim": 8},
        vision_config={"hidden_size": 32, "intermediate_size": 64, "num_heads": 4, "depth": 2,
                       "patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2,
                       "out_hidden_size": 32, "num_position_embeddings": 64,
                       "deepstack_visual_indexes": [0, 1]},
        image_token_id=5, video_token_id=6, vision_start_token_id=3, vision_end_token_id=4,
    )


@pytest.fixture(scope="module")
def tiny_encoder():
    from transformers import Qwen3VLModel

    torch.manual_seed(0)
    # requires_grad off, as a text encoder that is not being trained has it in a real run
    return Qwen3VLModel(tiny_config()).eval().requires_grad_(False)


class FakeTokenizer:
    eos_token_id = 159
    pad_token_id = None


class FakeProcessor:
    tokenizer = FakeTokenizer()


# --- the graft, measured --------------------------------------------------------------------------


def test_the_shell_is_exactly_the_encoder_plus_its_tied_head(tiny_encoder):
    shell = build_generation_shell(tiny_encoder, FakeProcessor())

    ids = torch.randint(7, 150, (1, 9))
    mask = torch.ones_like(ids)
    with torch.no_grad():
        hidden = tiny_encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        expected = hidden @ tiny_encoder.get_input_embeddings().weight.T
        actual = shell(input_ids=ids, attention_mask=mask).logits

    # exact, not approximate: the head IS the embedding matrix, so there is nothing to differ
    assert torch.equal(actual, expected)


def test_no_weights_are_copied(tiny_encoder):
    shell = build_generation_shell(tiny_encoder, FakeProcessor())

    assert shell.model is tiny_encoder
    assert shell.lm_head.weight.data_ptr() == tiny_encoder.get_input_embeddings().weight.data_ptr()


def test_nothing_is_left_on_the_meta_device(tiny_encoder):
    shell = build_generation_shell(tiny_encoder, FakeProcessor())

    assert all(p.device.type != "meta" for p in shell.parameters())
    assert all(b.device.type != "meta" for b in shell.buffers())


def test_the_shell_can_actually_generate(tiny_encoder):
    shell = build_generation_shell(tiny_encoder, FakeProcessor())

    ids = torch.randint(7, 150, (1, 9))
    with torch.no_grad():
        out = shell.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                             max_new_tokens=4, do_sample=False)

    assert out.shape[1] > ids.shape[1]


def test_an_untied_model_is_refused_not_garbled():
    from transformers import Qwen3VLModel

    untied = Qwen3VLModel(tiny_config(tie=False)).eval()

    # without the tie, the "head" would be a random matrix producing confident nonsense
    with pytest.raises(RuntimeError, match="tie"):
        build_generation_shell(untied, FakeProcessor())


def test_the_training_models_config_is_not_touched(tiny_encoder):
    before = tiny_encoder.config.to_dict()

    build_generation_shell(tiny_encoder, FakeProcessor())

    assert tiny_encoder.config.to_dict() == before


def test_the_stop_token_comes_from_the_tokenizer(tiny_encoder):
    shell = build_generation_shell(tiny_encoder, FakeProcessor())

    assert shell.generation_config.eos_token_id == 159


# --- the captioner wrapper ------------------------------------------------------------------------


def test_from_resident_loads_nothing(tiny_encoder):
    captioner = QwenVLModel.from_resident(tiny_encoder, torch.device("cpu"),
                                          processor=FakeProcessor())

    assert captioner.model.model is tiny_encoder
    assert captioner.dtype == torch.float32


def test_release_restores_the_encoders_training_mode(tiny_encoder):
    tiny_encoder.train()
    try:
        captioner = QwenVLModel.from_resident(tiny_encoder, torch.device("cpu"),
                                              processor=FakeProcessor())
        assert not tiny_encoder.training  # the shell put it in eval for generation

        captioner.release()

        assert tiny_encoder.training  # back the way the training loop had it
    finally:
        tiny_encoder.eval()


def test_release_runs_the_restore_callback(tiny_encoder):
    captioner = QwenVLModel.from_resident(tiny_encoder, torch.device("cpu"),
                                          processor=FakeProcessor())
    calls = []
    captioner.on_release(lambda: calls.append("restored"))

    captioner.release()

    assert calls == ["restored"]
    assert captioner.model is None


# --- the provider's judgement ---------------------------------------------------------------------


class FakeModelType:
    """Just enough of ModelType: the part list, denoiser first, as the real enum declares it."""

    @staticmethod
    def denoising_model_part():
        return "transformer"


class FakeKrea2Model:
    """Just enough of Krea2Model: a text_encoder and BaseModel's materialize()/evict() API,
    which replaced the per-model *_to(device) movers upstream."""

    model_type = FakeModelType()

    def __init__(self, text_encoder):
        self.text_encoder = text_encoder
        self.moves: list = []

    def materialize(self, *parts):
        self.moves.append(("materialize", parts))

    def evict(self, *parts):
        self.moves.append(("evict", parts))


def make_curation(tmp_path, model, monkeypatch) -> DatasetCuration:
    import modules.module.QwenVLModel as qwen_module

    monkeypatch.setattr(qwen_module.AutoProcessor, "from_pretrained",
                        classmethod(lambda cls, name, **kw: FakeProcessor()))
    config = make_config(curate_recaption=True,
                        curate_recaption_model=CurationCaptionModel.QWEN3_VL_4B)
    config.train_device = "cpu"
    config.temp_device = "cpu"
    curation = DatasetCuration(make_config() if model is None else config, str(tmp_path))
    if model is not None:
        curation.config = config
        curation.attach_model(model)
    return curation


def test_a_krea2_run_captions_with_its_own_text_encoder(tmp_path, tiny_encoder, monkeypatch):
    curation = make_curation(tmp_path, FakeKrea2Model(tiny_encoder), monkeypatch)

    captioner = curation._resident_qwen()

    assert captioner is not None
    assert captioner.model.model is tiny_encoder
    curation.close()


def test_without_an_attached_model_the_fresh_load_is_used(tmp_path, monkeypatch):
    curation = make_curation(tmp_path, None, monkeypatch)

    assert curation._resident_qwen() is None
    curation.close()


def test_a_model_without_a_qwen_text_encoder_is_not_borrowed(tmp_path, monkeypatch):
    class SdxlModel:
        text_encoder = torch.nn.Linear(4, 4)  # a CLIP stand-in: wrong class entirely

    curation = make_curation(tmp_path, SdxlModel(), monkeypatch)

    assert curation._resident_qwen() is None
    curation.close()


def test_an_encoder_being_trained_is_left_alone(tmp_path, monkeypatch):
    from transformers import Qwen3VLModel

    torch.manual_seed(1)
    trained = Qwen3VLModel(tiny_config()).eval()
    trained.requires_grad_(True)

    curation = make_curation(tmp_path, FakeKrea2Model(trained), monkeypatch)

    # captions written through a half-trained LoRA would be shaped by weights still moving
    assert curation._resident_qwen() is None
    curation.close()


def test_an_fp8_encoder_is_left_alone(tmp_path, tiny_encoder, monkeypatch):
    class Fp8Param:
        dtype = getattr(torch, "float8_e4m3fn", torch.int8)
        device = torch.device("cpu")

        def __init__(self):
            self.requires_grad = False

    class Fp8Encoder:
        config = tiny_encoder.config
        visual = object()

        def parameters(self):
            return iter([Fp8Param()])

    Fp8Encoder.__name__ = "Qwen3VLModel"
    curation = make_curation(tmp_path, FakeKrea2Model.__new__(FakeKrea2Model), monkeypatch)
    curation._model_ref = type("M", (), {"text_encoder": Fp8Encoder()})()

    # the fp8 vision tower cannot run; the quantized fresh load takes over instead
    assert curation._resident_qwen() is None
    curation.close()


def test_an_offloaded_encoder_is_not_moved_into_vram_that_is_not_there(tmp_path, tiny_encoder, monkeypatch):
    curation = make_curation(tmp_path, FakeKrea2Model(tiny_encoder), monkeypatch)
    curation.device = torch.device("cuda")  # pretend training runs on a card...
    monkeypatch.setattr("modules.module.QwenVLModel._free_vram_gb", lambda device: 0.5)

    assert curation._resident_qwen() is None  # ...with no room to bring the encoder back
    curation.close()


def test_moving_the_encoder_goes_through_the_models_own_mover(tmp_path, monkeypatch):
    from transformers import Qwen3VLModel

    torch.manual_seed(2)
    encoder = Qwen3VLModel(tiny_config()).eval().requires_grad_(False)
    holder = FakeKrea2Model(encoder)
    curation = make_curation(tmp_path, holder, monkeypatch)

    # the encoder is "offloaded": its device differs from the training device in name only, so the
    # move machinery runs without needing actual hardware
    monkeypatch.setattr(curation, "device", torch.device("cpu", 0))
    monkeypatch.setattr("modules.util.curation.DatasetCuration.DatasetCuration._room_to_move",
                        lambda self, parameters: True)

    captioner = curation._resident_qwen()

    # moved through BaseModel.materialize so a layer offload conductor stays consistent
    assert holder.moves and holder.moves[0] == ("materialize", ("text_encoder",))

    captioner.release()
    assert holder.moves[-1] == ("evict", ("text_encoder",))  # and put back afterwards
    curation.close()


def test_the_recaptioner_prefers_the_resident_encoder(tmp_path, tiny_encoder, monkeypatch):
    provided = QwenVLModel.from_resident(tiny_encoder, torch.device("cpu"),
                                         processor=FakeProcessor())
    recaptioner = Recaptioner(CurationCaptionModel.QWEN3_VL_4B, torch.device("cpu"),
                              resident_provider=lambda: provided)

    assert recaptioner._load_model() is provided


def test_when_there_is_no_resident_encoder_the_fresh_load_still_happens(monkeypatch):
    import modules.module.QwenVLModel as qwen_module

    loaded = {}
    monkeypatch.setattr(qwen_module.AutoProcessor, "from_pretrained",
                        classmethod(lambda cls, name, **kw: FakeProcessor()))
    monkeypatch.setattr(qwen_module.Qwen3VLForConditionalGeneration, "from_pretrained",
                        classmethod(lambda cls, name, **kw: loaded.setdefault("model", _FreshStub())))
    recaptioner = Recaptioner(CurationCaptionModel.QWEN3_VL_4B, torch.device("cpu"),
                              resident_provider=lambda: None)

    recaptioner._load_model()

    assert "model" in loaded  # fell through to the download path


class _FreshStub:
    def eval(self):
        return self

    def to(self, device):
        return self


def test_a_crashing_provider_does_not_take_recaptioning_down(tmp_path, monkeypatch):
    class Exploding:
        text_encoder = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))

    curation = make_curation(tmp_path, None, monkeypatch)
    curation._model_ref = Exploding()

    assert curation._resident_qwen() is None  # falls back, does not raise
    curation.close()


# --- the denoiser stepping aside ------------------------------------------------------------------
#
# On a card where neither the resident encoder nor even the quantized captioner fits beside the
# denoiser, the boundary used to be skipped. The denoiser is idle at a boundary, so instead it moves
# to system RAM for the duration and comes straight back.


class SwappableModel(FakeKrea2Model):
    """A Krea2Model stand-in — the denoiser moves through the same materialize/evict API."""

    @property
    def transformer_moves(self):
        return [m for m in self.moves if m[1] == ("transformer",)]


def cuda_curation(tmp_path, model, monkeypatch, free_gb):
    curation = make_curation(tmp_path, model, monkeypatch)
    curation.device = torch.device("cuda")
    monkeypatch.setattr("modules.module.QwenVLModel._free_vram_gb", lambda device: free_gb)
    return curation


def test_a_full_card_moves_the_denoiser_aside_for_the_boundary(tmp_path, tiny_encoder, monkeypatch):
    model = SwappableModel(tiny_encoder)
    # below even the tiny encoder's ~1.5GB margin, so the boundary genuinely has no room
    curation = cuda_curation(tmp_path, model, monkeypatch, free_gb=0.5)

    with curation._boundary_room():
        # the denoiser left before captioning started...
        assert model.transformer_moves == [("evict", ("transformer",))]

    # ...and was back before training resumed
    assert model.transformer_moves == [("evict", ("transformer",)), ("materialize", ("transformer",))]
    curation.close()


def test_a_card_with_room_leaves_the_denoiser_alone(tmp_path, tiny_encoder, monkeypatch):
    model = SwappableModel(tiny_encoder)
    curation = cuda_curation(tmp_path, model, monkeypatch, free_gb=64.0)

    with curation._boundary_room():
        pass

    assert model.transformer_moves == []
    curation.close()


def test_an_encoder_already_resident_needs_no_room_at_all(tmp_path, monkeypatch):
    from transformers import Qwen3VLModel

    torch.manual_seed(3)
    encoder = Qwen3VLModel(tiny_config()).eval().requires_grad_(False)
    model = SwappableModel(encoder)
    curation = cuda_curation(tmp_path, model, monkeypatch, free_gb=0.1)
    # the encoder "is" on the training device: make the comparison agree
    curation.device = next(encoder.parameters()).device

    assert curation._captioning_need_gb() == 0.0
    with curation._boundary_room():
        pass
    assert model.transformer_moves == []
    curation.close()


def test_the_denoiser_comes_back_even_when_captioning_blows_up(tmp_path, tiny_encoder, monkeypatch):
    model = SwappableModel(tiny_encoder)
    curation = cuda_curation(tmp_path, model, monkeypatch, free_gb=0.5)

    with pytest.raises(RuntimeError), curation._boundary_room():
        raise RuntimeError("captioning failed")

    assert model.transformer_moves[-1] == ("materialize", ("transformer",))
    curation.close()


def test_a_model_with_no_movable_denoiser_still_captions_as_before(tmp_path, tiny_encoder, monkeypatch):
    class BareModel:
        text_encoder = tiny_encoder  # no materialize/evict, no model_type

    model = BareModel()
    curation = cuda_curation(tmp_path, model, monkeypatch, free_gb=0.5)

    with curation._boundary_room():
        pass  # no crash: the downstream guards still protect the run

    curation.close()


def test_the_need_is_the_offloaded_encoders_own_size(tmp_path, tiny_encoder, monkeypatch):
    model = SwappableModel(tiny_encoder)
    curation = cuda_curation(tmp_path, model, monkeypatch, free_gb=2.2)

    weights_gb = sum(p.numel() * p.element_size() for p in tiny_encoder.parameters()) / 1024 ** 3

    assert curation._captioning_need_gb() == pytest.approx(weights_gb + 1.5)
    curation.close()


def test_an_unusable_encoder_falls_back_to_the_quantized_budget(tmp_path, monkeypatch):
    from modules.module.QwenVLModel import REQUIRED_FREE_GB
    from modules.util.enum.CurationCaptionPrecision import CurationCaptionPrecision

    class NotQwen:
        text_encoder = torch.nn.Linear(4, 4)

    curation = make_curation(tmp_path, None, monkeypatch)
    curation.config.curate_recaption_model = CurationCaptionModel.QWEN3_VL_4B
    curation.config.curate_recaption_precision = CurationCaptionPrecision.NF4
    curation._model_ref = NotQwen()

    assert curation._captioning_need_gb() == REQUIRED_FREE_GB[CurationCaptionPrecision.NF4]
    curation.close()


def test_a_crashing_need_estimate_does_not_stop_the_boundary(tmp_path, monkeypatch):
    curation = make_curation(tmp_path, None, monkeypatch)
    curation.device = torch.device("cuda")
    monkeypatch.setattr(curation, "_captioning_need_gb",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    with curation._boundary_room():
        pass  # falls through to the ordinary guarded path

    curation.close()
