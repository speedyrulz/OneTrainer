"""End-to-end test of the multi-config tournament against a stand-in model.

The model, data pipeline and saver are stand-ins, but everything that decides the outcome is the real
thing: ``MultiConfigTrainer`` drives ``GenericTrainer.train`` over real optimizers built by
``init_model_parameters``, real LR schedulers, and the real segment / promotion / snapshot logic.

The stand-in training problem is a quadratic - minimise ``mean((w - target) ** 2)`` with plain SGD -
so which learning rate should win is arithmetic rather than opinion. With a gradient of
``2 * (w - target) / n``, one step multiplies the error by ``|1 - 2 * lr / n|``: a tiny learning rate
barely moves, a well-chosen one shrinks it, and a large one diverges.

Run with:  pytest tests/test_multi_config_trainer.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mgds")
pytest.importorskip("diffusers")

from modules.trainer.GenericTrainer import GenericTrainer  # noqa: E402
from modules.trainer.MultiConfigTrainer import MultiConfigTrainer  # noqa: E402
from modules.util.callbacks.TrainCallbacks import TrainCallbacks  # noqa: E402
from modules.util.commands.TrainCommands import TrainCommands  # noqa: E402
from modules.util.config.TrainConfig import TrainConfig  # noqa: E402
from modules.util.enum.LearningRateScheduler import LearningRateScheduler  # noqa: E402
from modules.util.enum.MultiConfigMode import MultiConfigMode  # noqa: E402
from modules.util.enum.MultiConfigPruneStrategy import MultiConfigPruneStrategy  # noqa: E402
from modules.util.enum.MultiConfigStateStorage import MultiConfigStateStorage  # noqa: E402
from modules.util.enum.Optimizer import Optimizer  # noqa: E402
from modules.util.enum.TimeUnit import TimeUnit  # noqa: E402
from modules.util.NamedParameterGroup import NamedParameterGroup, NamedParameterGroupCollection  # noqa: E402
from modules.util.TrainProgress import TrainProgress  # noqa: E402

PARAM_SIZE = 8
STEPS_PER_EPOCH = 4
TARGET = 0.0


# --- stand-ins ---------------------------------------------------------------------------------


class FakeTensorboard:
    def __init__(self):
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, float(value), int(step)))

    def add_image(self, *_args, **_kwargs):
        pass

    def close(self):
        pass

    def tags(self) -> set[str]:
        return {tag for tag, _, _ in self.scalars}


class FakeModel:
    def __init__(self):
        self.build_parameters()

        self.optimizer = None
        self.optimizer_state_dict = None
        self.param_group_mapping = None
        self.ema = None
        self.ema_state_dict = None
        self.train_progress = TrainProgress()
        self.train_config = None

    def build_parameters(self):
        """Stands in for rebuilding an adapter: fresh weights in a fresh parameter group."""
        self.weight = torch.nn.Parameter(torch.full((PARAM_SIZE,), 4.0))

        collection = NamedParameterGroupCollection()
        collection.add_group(NamedParameterGroup(
            unique_name="weight", parameters=[self.weight], learning_rate=None
        ))
        self.parameters = collection

    def adapters(self):
        return []

    def evict(self, *parts):
        # BaseModel's API: no to(); parts move between train and temp devices
        return self

    def materialize(self, *parts):
        return self

    def eval(self):
        return self


class FakeModelSetup:
    def __init__(self, model: FakeModel):
        self.model = model
        self.setup_train_device_calls = 0

    def setup_train_device(self, _model, _config):
        self.setup_train_device_calls += 1

    def setup_model(self, model, config):
        from modules.util.optimizer_util import init_model_parameters

        model.train_config = config
        model.build_parameters()
        init_model_parameters(model, model.parameters, torch.device(config.train_device))

    def predict(self, _model, _batch, _config, _train_progress, deterministic=False):  # noqa: ARG002
        return {}

    def calculate_loss(self, _model, _batch, _output, _config):
        return ((self.model.weight - TARGET) ** 2).mean()

    def after_optimizer_step(self, _model, _config, _train_progress):
        pass

    def report_to_tensorboard(self, _model, _config, _lr_scheduler, _tensorboard):
        pass


class FakePipeline:
    modules = ()


class FakeDataSet:
    def __init__(self, batches: list[dict], start_index: int):
        self.batches = batches
        self.start_index = start_index
        self.epochs_started = 0
        self.loading_pipeline = FakePipeline()

    def start_next_epoch(self):
        self.epochs_started += 1

    def approximate_length(self):
        return len(self.batches)


class FakeDataLoader:
    """Serves a fixed number of batches per epoch, honouring a mid-epoch start offset like mgds."""

    def __init__(self, batches_per_epoch: int, start_index: int, is_validation: bool):
        if is_validation:
            batches = [{
                "concept_name": ["validation"],
                "concept_path": ["/validation"],
                "concept_seed": torch.tensor(7),
            } for _ in range(batches_per_epoch)]
        else:
            batches = [{"concept_type": ["STANDARD"]} for _ in range(batches_per_epoch)]

        self.data_set = FakeDataSet(batches, start_index)
        # only the first epoch served skips ahead, matching mgds' initial_epoch_sample behaviour
        self.first_epoch = True

    def get_data_set(self):
        return self.data_set

    def get_data_loader(self):
        if self.first_epoch:
            self.first_epoch = False
            return list(self.data_set.batches[self.data_set.start_index:])
        return list(self.data_set.batches)


class FakeModelSaver:
    def __init__(self):
        self.saves: list[str] = []

    def save(self, *_args, **kwargs):
        self.saves.append(kwargs.get("output_model_destination", "<positional>"))


class FakeLayerMixin:
    """Replaces the model, data and saver layer with stand-ins, and records what the trainer does."""

    def install_fakes(self):
        self.model = FakeModel()
        self.model.train_config = self.config
        self.model_setup = FakeModelSetup(self.model)
        self.model_saver = FakeModelSaver()
        self.model_sampler = None
        self.parameters = self.model.parameters.parameters()
        self.previous_sample_time = -1
        self.sample_queue = []

        from modules.util.optimizer_util import init_model_parameters
        init_model_parameters(self.model, self.model.parameters, self.train_device)

        self.data_loader = self.create_data_loader(self.model, self.model_setup, self.model.train_progress)
        self.validation_data_loader = self.create_data_loader(
            self.model, self.model_setup, self.model.train_progress, is_validation=True
        )

        self.segment_starts: list[tuple[int, str, float, int]] = []
        self.promoted_sampling_calls: list[int] = []
        self.all_sampling_calls: list[int] = []
        self.data_loader_builds: list[tuple[bool, int, int]] = []
        self.rebuild_calls: list[str] = []

    def create_data_loader(self, model, model_setup, train_progress, is_validation=False):  # noqa: ARG002
        start_index = 0 if is_validation else train_progress.epoch_step
        if hasattr(self, "data_loader_builds"):
            self.data_loader_builds.append((is_validation, train_progress.epoch, train_progress.epoch_step))
        return FakeDataLoader(STEPS_PER_EPOCH, start_index, is_validation)

    def _GenericTrainer__sample_during_training(self, train_progress, _train_device, _sample_params_list=None):
        # shadows the name-mangled private method, so this records sampling from *either* path: the
        # in-loop trigger inside the batch loop as well as the tournament's own call
        self.all_sampling_calls.append(train_progress.global_step)


class PlainHarnessTrainer(FakeLayerMixin, GenericTrainer):
    """Ordinary single-config training over the same stand-ins, to guard the shared trainer."""


class HarnessTrainer(FakeLayerMixin, MultiConfigTrainer):
    """MultiConfigTrainer over the same stand-ins, with the round transitions recorded."""

    def _begin_candidate(self, candidate, round_start):
        super()._begin_candidate(candidate, round_start)
        # record what each candidate actually starts its segment from. Recorded after the call,
        # because that is what puts the model into the state the segment begins from.
        self.segment_starts.append((
            self.round_index,
            candidate.name,
            float(self.model.weight[0].item()),
            self.model.train_progress.global_step,
        ))

    def _rebuild_model_for(self, candidate):
        self.rebuild_calls.append(candidate.name)
        return super()._rebuild_model_for(candidate)

    def run_sampling(self, train_progress):
        # the path the tournament uses after promoting a winner
        self.promoted_sampling_calls.append(train_progress.global_step)
        super().run_sampling(train_progress)


# --- fixtures ----------------------------------------------------------------------------------


def _write_candidate(tmp_path, name: str, learning_rate: float) -> str:
    config = TrainConfig.default_values()
    config.learning_rate = learning_rate
    config.learning_rate_scheduler = LearningRateScheduler.CONSTANT
    config.learning_rate_warmup_steps = 0
    # the 1.0 default turns the quadratic into constant-size steps and hides the learning rate
    config.clip_grad_norm = None  # the 200-step default would hide the learning rate here
    config.optimizer.optimizer = Optimizer.SGD
    path = os.path.join(str(tmp_path), f"{name}.json")
    with open(path, "w") as f:
        json.dump(config.to_settings_dict(secrets=False), f)
    return path


def _base_config(tmp_path, epochs: int, storage: MultiConfigStateStorage) -> TrainConfig:
    config = TrainConfig.default_values()
    config.workspace_dir = os.path.join(str(tmp_path), "workspace")
    config.cache_dir = os.path.join(str(tmp_path), "cache")
    config.epochs = epochs
    config.batch_size = 1
    config.gradient_accumulation_steps = 1
    config.validation = True
    config.validate_after = 1
    config.validate_after_unit = TimeUnit.EPOCH
    config.tensorboard = False
    config.optimizer.optimizer = Optimizer.SGD
    config.learning_rate_scheduler = LearningRateScheduler.CONSTANT
    config.learning_rate_warmup_steps = 0
    # the 1.0 default turns the quadratic into constant-size steps and hides the learning rate
    config.clip_grad_norm = None
    config.sample_after_unit = TimeUnit.NEVER
    config.save_every_unit = TimeUnit.NEVER
    config.backup_after_unit = TimeUnit.NEVER
    config.samples = []
    config.concepts = []

    config.multi_config = True
    config.multi_config_state_storage = storage
    # error factor per step = |1 - 2 * lr / PARAM_SIZE|:
    #   0.004 -> 0.999 (barely moves), 2.0 -> 0.5 (halves the error), 12.0 -> 2.0 (diverges)
    config.multi_config_path_1 = _write_candidate(tmp_path, "slow", 0.004)
    config.multi_config_path_2 = _write_candidate(tmp_path, "good", 2.0)
    config.multi_config_path_3 = _write_candidate(tmp_path, "divergent", 12.0)
    return config


def build_trainer(
        tmp_path,
        monkeypatch,
        epochs=3,
        storage=MultiConfigStateStorage.DISK,
        configure=None,
) -> HarnessTrainer:
    monkeypatch.setattr("modules.trainer.GenericTrainer.init_compile", lambda: None)
    monkeypatch.setattr("modules.trainer.GenericTrainer.SummaryWriter", lambda *_a, **_kw: FakeTensorboard())

    def fake_generic_start(self):
        self.install_fakes()

    monkeypatch.setattr(GenericTrainer, "start", fake_generic_start)

    config = _base_config(tmp_path, epochs, storage)
    if configure is not None:
        configure(config)
    trainer = HarnessTrainer(config, TrainCallbacks(), TrainCommands())
    trainer.start()
    return trainer


def adaptive(start_lr: float):
    """Turn the harness config into an adaptive learning rate tournament.

    With the stand-in quadratic, one SGD step multiplies the error by ``|1 - lr / 8 * 2|``, so a
    learning rate of 4.0 lands exactly on the target and anything either side of it is worse. The
    ladder should therefore climb towards 4.0 and stay there.
    """
    def configure(config):
        config.multi_config_mode = MultiConfigMode.ADAPTIVE_LEARNING_RATE
        config.multi_config_adaptive_lr = start_lr
    return configure


def end_early(enabled: bool = True):
    def configure(config):
        config.multi_config_end_early = enabled
    return configure


def combine(*configurers):
    def configure(config):
        for configurer in configurers:
            configurer(config)
    return configure


# --- tests -------------------------------------------------------------------------------------


@pytest.mark.parametrize("storage", [MultiConfigStateStorage.DISK, MultiConfigStateStorage.RAM])
def test_the_best_config_wins_every_round(tmp_path, monkeypatch, storage):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=3, storage=storage)
    trainer.train()

    assert len(trainer.history) == 3
    assert [entry['winner'] for entry in trainer.history] == ["good", "good", "good"]

    # the run ends up where the winning settings would have taken it: 12 steps of halving the error
    # from a starting value of 4.0
    assert trainer.model.weight[0].item() == pytest.approx(4.0 * 0.5 ** 12, rel=1e-3)


def test_every_config_in_a_round_starts_from_the_same_state(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=3)
    trainer.train()

    by_round: dict[int, list] = {}
    for round_index, name, weight, step in trainer.segment_starts:
        by_round.setdefault(round_index, []).append((name, weight, step))

    assert len(by_round) == 3
    for round_index, starts in by_round.items():
        assert [name for name, _, _ in starts] == ["slow", "good", "divergent"], \
            f"round {round_index} ran configs out of order"
        weights = {round(weight, 10) for _, weight, _ in starts}
        steps = {step for _, _, step in starts}
        assert len(weights) == 1, f"round {round_index} configs started from different weights"
        assert len(steps) == 1, f"round {round_index} configs started from different step counts"


def test_each_round_starts_from_the_previous_winner(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=3)
    trainer.train()

    starts = [weight for round_index, name, weight, _ in trainer.segment_starts if name == "slow"]
    # every round begins where the winner ("good", halving each of 4 steps) left off
    assert starts[0] == pytest.approx(4.0)
    assert starts[1] == pytest.approx(4.0 * 0.5 ** 4, rel=1e-3)
    assert starts[2] == pytest.approx(4.0 * 0.5 ** 8, rel=1e-3)


def test_every_config_trains_the_same_number_of_steps(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=3)
    trainer.train()

    for entry in trainer.history:
        step_counts = {result['steps_trained'] for result in entry['results']}
        assert step_counts == {STEPS_PER_EPOCH}, f"round {entry['round']} had uneven segments"


def test_progress_advances_exactly_one_interval_per_round(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=3)
    trainer.train()

    boundaries = [(entry['start_epoch'], entry['end_epoch'], entry['end_global_step'])
                  for entry in trainer.history]
    assert boundaries == [(0, 1, 4), (1, 2, 8), (2, 3, 12)]
    assert trainer.model.train_progress.epoch == 3
    assert trainer.finished_reason == "reached the configured epoch count"


def test_a_diverging_config_never_wins(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=3)
    trainer.train()

    for entry in trainer.history:
        scores = {result['candidate']: result['score'] for result in entry['results']}
        assert scores['divergent'] > scores['good']
        assert scores['slow'] > scores['good']


def test_results_and_summary_are_written(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=2)
    trainer.train()
    trainer.end()

    multi_config_dir = os.path.join(trainer.config.workspace_dir, "multi_config")

    with open(os.path.join(multi_config_dir, "candidates.json")) as f:
        candidates = json.load(f)
    assert [c['name'] for c in candidates['candidates']] == ["slow", "good", "divergent"]
    assert candidates['segment'] == "1 epoch(s)"

    with open(os.path.join(multi_config_dir, "results.json")) as f:
        results = json.load(f)
    assert len(results) == 2
    assert set(results[0]['results'][0].keys()) >= {'candidate', 'score', 'validation', 'steps_trained'}

    with open(os.path.join(multi_config_dir, "summary.json")) as f:
        summary = json.load(f)
    assert summary['rounds'] == 2
    assert summary['wins'] == {"slow": 0, "good": 2, "divergent": 0}


def test_each_config_gets_its_own_tensorboard_series(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=2)
    trainer.train()

    tags = trainer.tensorboard.tags()
    assert "loss/multi_config/slow/validation" in tags
    assert "loss/multi_config/good/validation" in tags
    assert "loss/multi_config/divergent/validation" in tags
    assert "loss/multi_config/winner" in tags
    # the winner is also logged under the normal series so the usual curve stays continuous
    assert "loss/validation_step/validation" in tags


def test_the_winning_model_is_saved_once_per_round(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=3)
    trainer.train()

    assert len(trainer.model_saver.saves) == 3
    for round_index, path in enumerate(trainer.model_saver.saves, start=1):
        assert f"round{round_index:03d}-good-" in os.path.basename(path)


def test_saving_each_round_can_be_turned_off(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=2)
    trainer.base_config.multi_config_save_each_round = False
    trainer.train()

    assert trainer.model_saver.saves == []


def test_sampling_runs_once_per_round_not_once_per_config(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=3)
    # would fire on every single step if the tournament did not suppress the in-loop trigger
    trainer.base_config.sample_after_unit = TimeUnit.STEP
    trainer.base_config.sample_after = 1
    for candidate in trainer.candidates:
        candidate.config.sample_after_unit = TimeUnit.STEP
        candidate.config.sample_after = 1
    trainer.config.sample_after_unit = TimeUnit.STEP
    trainer.config.sample_after = 1

    trainer.train()

    # once per round, on the promoted model
    assert trainer.promoted_sampling_calls == [4, 8, 12]
    # and nothing else sampled: an unsuppressed per-step trigger would have fired on all 36
    # candidate steps instead
    assert trainer.all_sampling_calls == [4, 8, 12]


def test_stopping_mid_round_promotes_the_configs_that_finished(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=5)

    original = trainer._run_candidate_segment
    seen = []

    def stop_after_two(candidate, round_start):
        result = original(candidate, round_start)
        seen.append(candidate.name)
        if len(seen) == 2:
            trainer.commands.stop()
        return result

    trainer._run_candidate_segment = stop_after_two
    trainer.train()

    # the two configs that completed a segment were compared, and the better one was promoted
    assert len(trainer.history) == 1
    assert trainer.history[0]['winner'] == "good"
    assert [r['candidate'] for r in trainer.history[0]['results']] == ["slow", "good"]
    assert trainer.finished_reason == "stopped by user"


def test_stopping_before_any_config_finishes_rewinds_to_the_last_promoted_state(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=5)

    original = trainer._run_candidate_segment
    seen = []

    def stop_at_start_of_second_round(candidate, round_start):
        if len(seen) == 3:
            trainer.commands.stop()
        result = original(candidate, round_start)
        seen.append(candidate.name)
        return result

    trainer._run_candidate_segment = stop_at_start_of_second_round
    trainer.train()

    assert len(trainer.history) == 1
    # the interrupted fourth segment was discarded: we are back on the round-1 winner's state
    assert trainer.model.weight[0].item() == pytest.approx(4.0 * 0.5 ** 4, rel=1e-3)
    assert trainer.model.train_progress.global_step == 4


def test_data_pipeline_is_rebuilt_for_every_config(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=2)
    trainer.data_loader_builds.clear()
    trainer.train()

    train_builds = [b for b in trainer.data_loader_builds if not b[0]]
    validation_builds = [b for b in trainer.data_loader_builds if b[0]]

    # 3 configs x 2 rounds, each repositioned at the start of its round
    assert len(train_builds) == 6
    assert [b[1] for b in train_builds] == [0, 0, 0, 1, 1, 1]
    assert {b[2] for b in train_builds} == {0}

    assert len(validation_builds) == 6


def test_deterministic_data_off_skips_the_rebuild_on_epoch_boundaries(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=2)
    trainer.base_config.multi_config_deterministic_data = False
    trainer.data_loader_builds.clear()
    trainer.train()

    assert trainer.data_loader_builds == []


def test_step_based_intervals_produce_mid_epoch_rounds(tmp_path, monkeypatch):
    monkeypatch.setattr("modules.trainer.GenericTrainer.init_compile", lambda: None)
    monkeypatch.setattr("modules.trainer.GenericTrainer.SummaryWriter", lambda *_a, **_kw: FakeTensorboard())
    monkeypatch.setattr(GenericTrainer, "start", lambda self: self.install_fakes())

    config = _base_config(tmp_path, epochs=2, storage=MultiConfigStateStorage.DISK)
    config.validate_after = 2
    config.validate_after_unit = TimeUnit.STEP

    trainer = HarnessTrainer(config, TrainCallbacks(), TrainCommands())
    trainer.start()
    trainer.train()

    # 2 epochs x 4 steps = 8 steps, decided every 2 steps
    assert [entry['end_global_step'] for entry in trainer.history] == [2, 4, 6, 8]
    for entry in trainer.history:
        assert {r['steps_trained'] for r in entry['results']} == {2}

    # a mid-epoch round is repositioned inside the epoch, not restarted from its beginning
    mid_epoch_builds = [b for b in trainer.data_loader_builds if not b[0] and b[2] != 0]
    assert mid_epoch_builds, "expected at least one mid-epoch repositioning"


def make_independent(trainer):
    """Turn the tournament into the mode a layer filter sweep uses."""
    for candidate in trainer.candidates:
        candidate.requires_model_rebuild = True
    trainer.independent_lineages = True


def test_independent_lineages_keep_each_config_on_its_own_weights(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=2)
    make_independent(trainer)
    trainer.train()

    starts = {}
    for _round_index, name, weight, _step in trainer.segment_starts:
        starts.setdefault(name, []).append(weight)

    # Round 2 continues each config's own training instead of the round winner's. "slow" barely moves,
    # so it comes back to almost 4.0; under shared state it would have restarted from "good"'s 0.25.
    assert starts["slow"][0] == pytest.approx(4.0)
    assert starts["slow"][1] == pytest.approx(4.0 * 0.999 ** 4, rel=1e-3)
    assert starts["good"][1] == pytest.approx(4.0 * 0.5 ** 4, rel=1e-3)
    assert starts["divergent"][1] == pytest.approx(4.0 * 2 ** 4, rel=1e-3)


def test_independent_lineages_still_report_and_save_a_winner(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=2)
    make_independent(trainer)
    trainer.train()

    assert [entry['winner'] for entry in trainer.history] == ["good", "good"]
    assert len(trainer.model_saver.saves) == 2
    # the model left behind is the winner's, carried through its own lineage
    assert trainer.model.weight[0].item() == pytest.approx(4.0 * 0.5 ** 8, rel=1e-3)


def test_independent_lineages_rebuild_the_model_for_every_config(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=2)
    make_independent(trainer)
    trainer.rebuild_calls.clear()
    trainer.train()

    # once per config per round, plus once per round to put the winner back in place
    assert trainer.rebuild_calls == [
        "slow", "good", "divergent", "good",
        "slow", "good", "divergent", "good",
    ]


def test_shared_state_mode_never_rebuilds_the_model(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=2)
    trainer.rebuild_calls.clear()
    trainer.train()

    assert trainer.rebuild_calls == []


# --- adaptive learning rate -----------------------------------------------------------------------


def test_the_learning_rate_follows_the_winner_up_the_ladder(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=3, configure=adaptive(1.0))
    trainer.train()

    # 4.0 is the best possible rate here, so the ladder should walk towards it one rung per round
    assert [entry['winner'] for entry in trainer.history] == ["2", "3", "4"]
    assert trainer.ladder.centre == 4.0


def test_each_adaptive_round_compares_the_centre_and_its_neighbours(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=3, configure=adaptive(1.0))
    trainer.train()

    compared = [[result['candidate'] for result in entry['results']] for entry in trainer.history]
    assert compared == [
        ["0.9", "1", "2"],
        ["1", "2", "3"],
        ["2", "3", "4"],
    ]


def test_adaptive_rounds_all_start_from_the_promoted_state(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=2, configure=adaptive(1.0))
    trainer.train()

    by_round = {}
    for round_index, _name, weight, _step in trainer.segment_starts:
        by_round.setdefault(round_index, []).append(weight)

    for round_index, weights in by_round.items():
        assert len({round(w, 10) for w in weights}) == 1, \
            f"round {round_index} learning rates started from different weights"

    # round 2 continues from what lr=2.0 reached in round 1: four steps of halving the error
    assert by_round[1][0] == pytest.approx(4.0 * 0.5 ** 4, rel=1e-3)


def test_an_adaptive_start_between_rungs_is_snapped(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=1, configure=adaptive(1.4))

    assert trainer.ladder.centre == 1.0
    assert trainer.ladder.snapped_from == 1.4


def test_the_adaptive_summary_records_where_the_learning_rate_ended(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=3, configure=adaptive(1.0))
    trainer.train()
    trainer.end()

    with open(os.path.join(trainer.config.workspace_dir, "multi_config", "summary.json")) as f:
        summary = json.load(f)

    assert summary['final_learning_rate'] == 4.0
    assert summary['wins'] == {"2": 1, "3": 1, "4": 1}


# --- best model and ending early --------------------------------------------------------------------


def test_the_best_round_is_tracked(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=3)
    trainer.train()

    # the loss falls every round here, so the last round is the best one
    assert trainer.best_round == 3
    assert trainer.best_candidate.name == "good"
    assert trainer.best_score == pytest.approx(
        min(r['score'] for entry in trainer.history for r in entry['results'])
    )
    assert trainer.rounds_without_improvement == 0


def test_end_early_stops_once_the_loss_stops_falling(tmp_path, monkeypatch):
    # starting on 4.0 reaches the target in the first round, after which no round can do better
    trainer = build_trainer(
        tmp_path, monkeypatch, epochs=8, configure=combine(adaptive(4.0), end_early(True))
    )
    trainer.train()

    assert trainer.best_round == 1
    assert len(trainer.history) == 3, "one good round plus the two flat ones it waits for"
    assert "stopped early" in trainer.finished_reason
    assert trainer.model.train_progress.epoch < 8, "training should not have run to the end"


def test_end_early_off_keeps_training_through_flat_rounds(tmp_path, monkeypatch):
    trainer = build_trainer(
        tmp_path, monkeypatch, epochs=5, configure=combine(adaptive(4.0), end_early(False))
    )
    trainer.train()

    assert len(trainer.history) == 5
    assert trainer.finished_reason == "reached the configured epoch count"


def test_end_early_applies_to_the_other_modes_too(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=8, configure=end_early(True))
    # every config keeps improving on this problem, so nothing should trigger
    trainer.train()
    assert trainer.finished_reason == "reached the configured epoch count"

    trainer = build_trainer(tmp_path, monkeypatch, epochs=8, configure=end_early(True))
    make_independent(trainer)
    trainer.train()
    assert trainer.finished_reason == "reached the configured epoch count"


def test_end_early_patience_is_configurable(tmp_path, monkeypatch):
    def four_rounds_of_patience(config):
        config.multi_config_end_early = True
        config.multi_config_end_early_rounds = 4

    trainer = build_trainer(
        tmp_path, monkeypatch, epochs=12, configure=combine(adaptive(4.0), four_rounds_of_patience)
    )
    trainer.train()

    assert trainer.best_round == 1
    assert len(trainer.history) == 5, "one good round plus the four flat ones it now waits for"
    assert "stopped early" in trainer.finished_reason


def test_a_patience_of_one_stops_at_the_first_flat_round(tmp_path, monkeypatch):
    def impatient(config):
        config.multi_config_end_early = True
        config.multi_config_end_early_rounds = 1

    trainer = build_trainer(
        tmp_path, monkeypatch, epochs=12, configure=combine(adaptive(4.0), impatient)
    )
    trainer.train()

    assert len(trainer.history) == 2


def test_a_nonsense_patience_falls_back_to_the_default(tmp_path, monkeypatch):
    def broken(config):
        config.multi_config_end_early = True
        config.multi_config_end_early_rounds = 0

    trainer = build_trainer(tmp_path, monkeypatch, epochs=12, configure=combine(adaptive(4.0), broken))
    assert trainer.end_early_patience == 2


# --- dropping configs that stop winning -------------------------------------------------------------


def prune(after: int, strategy: MultiConfigPruneStrategy = MultiConfigPruneStrategy.NEVER_WON):
    def configure(config):
        config.multi_config_prune = True
        config.multi_config_prune_after = after
        config.multi_config_prune_strategy = strategy
    return configure


def prune_worst(after: int):
    return prune(after, MultiConfigPruneStrategy.WORST_AVERAGE)


def test_configs_that_win_nothing_in_the_window_are_dropped(tmp_path, monkeypatch):
    # "good" wins every round here, so the other two should go at the first check
    trainer = build_trainer(tmp_path, monkeypatch, epochs=6, configure=prune(2))
    trainer.train()

    assert [r['candidate'] for r in trainer.history[0]['results']] == ["slow", "good", "divergent"]
    assert [r['candidate'] for r in trainer.history[1]['results']] == ["slow", "good", "divergent"]
    # after two rounds without a win, only the winner is left
    assert [r['candidate'] for r in trainer.history[2]['results']] == ["good"]
    assert [c.name for c in trainer.candidates] == ["good"]


def test_pruning_never_drops_the_last_config(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=8, configure=prune(1))
    trainer.train()

    assert [c.name for c in trainer.candidates] == ["good"]
    # training carries on normally with the one that is left
    assert len(trainer.history) == 8
    assert trainer.finished_reason == "reached the configured epoch count"


def test_pruning_keeps_everything_that_won_inside_the_window(tmp_path, monkeypatch):
    # a window of 3 with 3 configs, each handed exactly one win: nothing has been idle, so the check
    # at the end of the window should drop nothing
    trainer = build_trainer(tmp_path, monkeypatch, epochs=3, configure=prune(3))

    winners = iter(["slow", "good", "divergent"])
    original_promote = trainer._promote

    def promote_in_turn(winner, results, start_epoch, start_step):
        wanted = next(winners, winner.candidate.name)
        chosen = next((r for r in results if r.candidate.name == wanted), winner)
        return original_promote(chosen, results, start_epoch, start_step)

    trainer._promote = promote_in_turn
    trainer.train()

    assert {c.name for c in trainer.candidates} == {"slow", "good", "divergent"}
    for entry in trainer.history:
        assert len(entry['results']) == 3


def test_pruning_is_off_by_default(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=6)
    trainer.train()

    assert len(trainer.candidates) == 3


def test_pruning_does_not_apply_to_the_adaptive_mode(tmp_path, monkeypatch):
    # every adaptive round compares a fresh window, so "has not won" says nothing about a rate
    trainer = build_trainer(
        tmp_path, monkeypatch, epochs=4, configure=combine(adaptive(1.0), prune(1))
    )
    trainer.train()

    for entry in trainer.history:
        assert len(entry['results']) == 3


def test_the_worst_average_rule_drops_one_config_at_a_time(tmp_path, monkeypatch):
    # losses here run good < slow < divergent, so the field narrows from the bottom up
    trainer = build_trainer(tmp_path, monkeypatch, epochs=6, configure=prune_worst(2))
    trainer.train()

    assert [r['candidate'] for r in trainer.history[1]['results']] == ["slow", "good", "divergent"]
    # after the first window only the worst is gone, not everything that failed to win
    assert [r['candidate'] for r in trainer.history[2]['results']] == ["slow", "good"]
    assert [r['candidate'] for r in trainer.history[4]['results']] == ["good"]


def test_the_worst_average_rule_keeps_a_close_second(tmp_path, monkeypatch):
    # "slow" never wins a round, so the never-won rule drops it immediately; on average loss it is
    # still ahead of "divergent" and survives the first window
    by_wins = build_trainer(tmp_path, monkeypatch, epochs=3, configure=prune(2))
    by_wins.train()
    assert [c.name for c in by_wins.candidates] == ["good"]

    by_average = build_trainer(tmp_path, monkeypatch, epochs=3, configure=prune_worst(2))
    by_average.train()
    assert [c.name for c in by_average.candidates] == ["slow", "good"]


def test_the_worst_average_rule_never_drops_the_last_config(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=10, configure=prune_worst(1))
    trainer.train()

    assert [c.name for c in trainer.candidates] == ["good"]
    assert len(trainer.history) == 10
    assert trainer.finished_reason == "reached the configured epoch count"


def test_the_worst_average_rule_ranks_on_every_round_so_far(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=2, configure=prune_worst(2))
    trainer.train()

    averages = {}
    for entry in trainer.history:
        for result in entry['results']:
            averages.setdefault(result['candidate'], []).append(result['score'])
    means = {name: sum(scores) / len(scores) for name, scores in averages.items()}

    dropped = {"slow", "good", "divergent"} - {c.name for c in trainer.candidates}
    assert dropped == {max(means, key=means.get)}


def test_the_average_covers_the_whole_run_not_just_the_last_check(tmp_path, monkeypatch):
    """A config's record must not be wiped clean every time the field is narrowed.

    With a check every round, ranking on the window alone would be ranking on the current round's
    loss - so a config that had been far behind for the whole run could survive on one good round.
    """
    trainer = build_trainer(tmp_path, monkeypatch, epochs=4, configure=prune_worst(1))

    seen_rounds = []
    original = trainer._survivors_by_average

    def record_and_run():
        seen_rounds.append({
            name: len(scores) for name, scores in trainer._candidate_scores.items()
        })
        return original()

    trainer._survivors_by_average = record_and_run
    trainer.train()

    # a check runs after each of the first two rounds; the third leaves one config, which is never
    # dropped, so no further check happens
    assert len(seen_rounds) == 2
    assert seen_rounds[0] == {"slow": 1, "good": 1, "divergent": 1}
    # the second check judges on two rounds, not on the one since the last check
    assert seen_rounds[1]["good"] == 2


def test_one_good_round_does_not_rescue_a_config_with_a_bad_record(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=3, configure=prune_worst(3))

    # give "divergent" a single excellent round right before the check; its earlier rounds should
    # still weigh it down enough to be dropped
    original_segment = trainer._run_candidate_segment

    def flatter_divergent(candidate, round_start):
        result = original_segment(candidate, round_start)
        if candidate.name == "divergent" and trainer.round_index == 2:
            result.score = 0.0
        return result

    trainer._run_candidate_segment = flatter_divergent
    trainer.train()

    assert "divergent" not in {c.name for c in trainer.candidates}


def test_a_config_with_no_validation_loss_is_dropped_first(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=2, configure=prune_worst(2))

    # "good" has the lowest loss but produces no usable score, so it cannot be ranked and goes first
    original_segment = trainer._run_candidate_segment

    def blank_out_good(candidate, round_start):
        result = original_segment(candidate, round_start)
        if candidate.name == "good":
            result.score = None
        return result

    trainer._run_candidate_segment = blank_out_good
    trainer.train()

    assert "good" not in {c.name for c in trainer.candidates}


def test_the_prune_rule_defaults_to_never_won():
    from modules.util.config.TrainConfig import TrainConfig

    assert TrainConfig.default_values().multi_config_prune_strategy \
        == MultiConfigPruneStrategy.NEVER_WON


def test_the_saved_model_is_the_best_one_not_the_last(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=3)

    # pin the best to round 1, so the run finishes somewhere the tournament does not consider best
    original_record = trainer._record_best

    def record_first_round_only(winner):
        if trainer.round_index == 0:
            original_record(winner)

    trainer._record_best = record_first_round_only
    trainer.train()

    # training carried on to round 3
    assert trainer.model.weight[0].item() == pytest.approx(4.0 * 0.5 ** 12, rel=1e-3)
    assert trainer.best_round == 1

    trainer.end()

    # ...but what is left for the final save is round 1's model
    assert trainer.model.weight[0].item() == pytest.approx(4.0 * 0.5 ** 4, rel=1e-3)


def test_nothing_is_restored_when_the_last_round_was_the_best(tmp_path, monkeypatch):
    trainer = build_trainer(tmp_path, monkeypatch, epochs=3)
    trainer.train()

    final_weight = trainer.model.weight[0].item()
    trainer.end()

    assert trainer.model.weight[0].item() == pytest.approx(final_weight)


def test_ordinary_single_config_training_is_unaffected(tmp_path, monkeypatch):
    """Guards the shared trainer: the tournament hooks must be inert when the tournament is off."""
    monkeypatch.setattr("modules.trainer.GenericTrainer.init_compile", lambda: None)
    monkeypatch.setattr("modules.trainer.GenericTrainer.SummaryWriter", lambda *_a, **_kw: FakeTensorboard())

    config = _base_config(tmp_path, epochs=3, storage=MultiConfigStateStorage.DISK)
    config.multi_config = False
    config.learning_rate = 2.0
    config.sample_after_unit = TimeUnit.EPOCH
    config.sample_after = 1

    trainer = PlainHarnessTrainer(config, TrainCallbacks(), TrainCommands())
    trainer.install_fakes()
    trainer.train()

    # every epoch ran, nothing was rewound
    assert trainer.model.train_progress.epoch == 3
    assert trainer.model.train_progress.global_step == 3 * STEPS_PER_EPOCH
    assert trainer.model.weight[0].item() == pytest.approx(4.0 * 0.5 ** 12, rel=1e-3)

    # the built-in triggers still fire: validation on its own schedule, sampling on its own
    assert "loss/validation_step/validation" in trainer.tensorboard.tags()
    assert trainer.all_sampling_calls, "scheduled sampling should still run outside the tournament"

    # and the segment machinery stayed out of the way
    assert trainer.segment_end_check is None
    assert trainer.segment_ended is False
    assert trainer.suppress_scheduled_validation is False
    assert trainer.suppress_scheduled_actions is False


def test_a_config_that_changes_optimizer_starts_with_fresh_optimizer_state(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("modules.trainer.GenericTrainer.init_compile", lambda: None)
    monkeypatch.setattr("modules.trainer.GenericTrainer.SummaryWriter", lambda *_a, **_kw: FakeTensorboard())
    monkeypatch.setattr(GenericTrainer, "start", lambda self: self.install_fakes())

    config = _base_config(tmp_path, epochs=1, storage=MultiConfigStateStorage.DISK)

    other = TrainConfig.default_values()
    other.learning_rate = 2.0
    other.learning_rate_scheduler = LearningRateScheduler.CONSTANT
    other.optimizer.optimizer = Optimizer.ADAMW
    adamw_path = os.path.join(str(tmp_path), "adamw.json")
    with open(adamw_path, "w") as f:
        json.dump(other.to_settings_dict(secrets=False), f)
    config.multi_config_path_3 = adamw_path

    trainer = HarnessTrainer(config, TrainCallbacks(), TrainCommands())
    trainer.start()
    trainer.train()

    assert len(trainer.history) == 1
    assert "fresh optimizer state" in capsys.readouterr().out
