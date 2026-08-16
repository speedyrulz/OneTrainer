"""Trains several candidate configs against each other, one validation interval at a time.

Each round:

1. every candidate is restored to the same shared training state,
2. trained for exactly one validation interval with its own settings,
3. validated on the same validation data,
4. the best validation loss wins, its state becomes the new shared state, and the model is saved.

The next round continues from the winner, so the run follows whichever settings are ahead at that
point in training rather than committing to one config up front.

Everything that decides the shape of the model, the data pipeline or the latent cache is taken from
the base config and shared by all candidates - see ``modules.util.multi_config_util`` for the list of
settings a candidate is actually allowed to change.
"""

import contextlib
import json
import math
import os
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.TrainConfig import MULTI_CONFIG_MIN_CANDIDATES, TrainConfig
from modules.util.enum.MultiConfigMode import MultiConfigMode
from modules.util.enum.MultiConfigPruneStrategy import MultiConfigPruneStrategy
from modules.util.enum.MultiConfigStateStorage import MultiConfigStateStorage
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.multi_config_ladder import LearningRateLadder, format_learning_rate
from modules.util.multi_config_util import (
    CandidateConfig,
    SegmentSchedule,
    build_ladder_candidates,
    load_candidate_configs,
    pick_winner,
    preflight_errors,
    score_validation_result,
)
from modules.util.optimizer_util import init_model_parameters
from modules.util.time_util import get_string_timestamp
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress
from modules.util.ValidationResult import ValidationResult

import torch

from tqdm import tqdm

BASE_STATE_KEY = "base"

# where the lowest-scoring state of the whole run is kept, so it can be restored for the final save
BEST_STATE_KEY = "best"

# fallback for how many rounds in a row may fail to improve on the best validation score before
# "End Early" stops the run, when the config asks for something nonsensical
DEFAULT_END_EARLY_PATIENCE = 2

# Fixed seed used when a config's adapter is built from scratch, so a run is reproducible and every
# config's fresh weights are drawn from the same starting point.
ADAPTER_INIT_SEED = 1234567


@dataclass
class CandidateRoundResult:
    candidate: CandidateConfig
    score: float | None
    validation: ValidationResult | None
    steps_trained: int
    segment_completed: bool
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            'candidate': self.candidate.name,
            'config_path': self.candidate.path,
            'score': self.score,
            'validation': self.validation.to_dict() if self.validation is not None else None,
            'steps_trained': self.steps_trained,
            'segment_completed': self.segment_completed,
            'error': self.error,
        }


def _detach_to_cpu(value):
    """Deep-copy a state structure onto the CPU.

    Optimizer and EMA state dicts hand back live references, so anything kept as a snapshot has to be
    cloned - otherwise the "snapshot" keeps changing as training continues. ``copy=True`` matters even
    when a tensor is already on the CPU, where ``.to("cpu")`` would otherwise return the tensor itself.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: _detach_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detach_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_to_cpu(item) for item in value)
    with contextlib.suppress(Exception):
        return deepcopy(value)
    return value


def _release_data_loader(data_loader):
    """Shut down the thread pool behind a data loader that is about to be dropped.

    Every mgds pipeline owns a ThreadPoolExecutor. The tournament builds a new pipeline per candidate
    segment, so without this the pools accumulate for the whole run.
    """
    if data_loader is None:
        return
    try:
        pipeline = data_loader.get_data_set().loading_pipeline
        states = {}
        for module in pipeline.modules:
            state = getattr(module, "_state", None)
            if state is not None:
                states[id(state)] = state
        for state in states.values():
            state.executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        # never let cleanup break a training run
        pass


class MultiConfigTrainer(GenericTrainer):
    def __init__(self, config: TrainConfig, callbacks: TrainCallbacks, commands: TrainCommands):
        super().__init__(config, callbacks, commands)

        # the settings currently open in the UI. Supplies everything the candidates are not allowed
        # to change, and stays untouched while self.config is swapped between candidates.
        self.base_config = config

        self.candidates: list[CandidateConfig] = []
        self.schedule = SegmentSchedule(config.validate_after, config.validate_after_unit)

        # A layer filter tournament changes which weights exist, so its candidates cannot hand their
        # state to one another: a narrower candidate has nowhere to put the layers a wider winner
        # trained, and OneTrainer's placeholder modules hold such weights without applying them to the
        # model. Those tournaments therefore run as independent lineages - every candidate continues
        # its own training and the rounds only decide which one is reported and saved.
        self.independent_lineages = False

        # set for the adaptive learning rate mode, where the candidates change every round
        self.ladder: LearningRateLadder | None = None

        self.round_index = 0
        self.history: list[dict] = []
        self.finished_reason: str | None = None

        # the lowest validation score of the whole run, the round that produced it, and how many
        # rounds have passed without beating it
        self.best_score: float | None = None
        self.best_round: int | None = None
        self.best_candidate: CandidateConfig | None = None
        self.rounds_without_improvement = 0

        # Wins inside the current pruning window, and how far through it we are. Wins reset at every
        # check because "has not won recently" is the question that rule asks.
        self._window_wins: dict[str, int] = {}
        self._rounds_since_prune = 0

        # every validation score each config has ever produced. Unlike the wins these are never
        # reset: the worst-average rule ranks on a config's whole record, so that one bad round early
        # on stays part of its average and one good round late does not erase it.
        self._candidate_scores: dict[str, list[float]] = {}

        # the config that produced the state stored under BASE_STATE_KEY: the base config until the
        # first round is decided, then whichever candidate last won
        self._promoted_config: TrainConfig = config
        self._promoted_candidate: CandidateConfig | None = None

        # progress at the last sample taken of a promoted model, so the user's sampling interval is
        # honoured once per round instead of once per candidate
        self._last_sample_epoch = 0
        self._last_sample_step = 0

        self.multi_config_dir = os.path.join(config.workspace_dir, "multi_config")
        self.state_dir = os.path.join(self.multi_config_dir, "state")
        self._ram_states: dict[str, dict] = {}

        # set by _restore_state, consumed by _activate_config
        self._pending_optimizer_state: dict | None = None
        self._pending_optimizer_name: str | None = None
        self._pending_ema_state: dict | None = None

    # --- setup ---------------------------------------------------------------------------------

    def start(self):
        errors = preflight_errors(self.base_config)
        if errors:
            raise RuntimeError("Multi-config training cannot start:\n  - " + "\n  - ".join(errors))

        if self.base_config.multi_config_mode == MultiConfigMode.ADAPTIVE_LEARNING_RATE:
            self.ladder = LearningRateLadder(
                self.base_config.multi_config_adaptive_lr,
                self.base_config.multi_config_adaptive_count,
            )
            if self.ladder.snapped_from is not None:
                tqdm.write(
                    f"Starting learning rate {self.ladder.snapped_from:g} is not a rung of the "
                    f"ladder; using {format_learning_rate(self.ladder.centre)} instead."
                )

        self.candidates, load_errors = load_candidate_configs(self.base_config)
        if load_errors:
            raise RuntimeError("Multi-config training cannot start:\n  - " + "\n  - ".join(load_errors))
        if len(self.candidates) < MULTI_CONFIG_MIN_CANDIDATES:
            raise RuntimeError(
                f"Multi-config training needs at least {MULTI_CONFIG_MIN_CANDIDATES} usable config sets, "
                f"got {len(self.candidates)}."
            )

        self.independent_lineages = any(candidate.requires_model_rebuild for candidate in self.candidates)

        os.makedirs(Path(self.multi_config_dir).absolute(), exist_ok=True)
        if self.base_config.multi_config_state_storage == MultiConfigStateStorage.DISK:
            os.makedirs(Path(self.state_dir).absolute(), exist_ok=True)
            # states from an earlier run belong to a different model and would never be read anyway;
            # dropping them keeps the directory honest and reclaims the disk
            self._clear_state_dir()

        super().start()

        self._write_candidate_report()

    def _write_candidate_report(self):
        report = {
            'created': get_string_timestamp(),
            'selection': str(self.base_config.multi_config_selection),
            'segment': self.schedule.describe(),
            'candidates': [
                {
                    'name': candidate.name,
                    'config_path': candidate.path,
                    'learning_rate': candidate.config.learning_rate,
                    'optimizer': str(candidate.config.optimizer.optimizer),
                    'learning_rate_scheduler': str(candidate.config.learning_rate_scheduler),
                    'ignored_fields': candidate.ignored_fields,
                }
                for candidate in self.candidates
            ],
        }

        if self.ladder is not None:
            report['learning_rate_ladder'] = {
                'start': self.ladder.centre,
                'first_round': self.ladder.values(),
            }

        tqdm.write("")
        tqdm.write("=" * 78)
        if self.ladder is not None:
            tqdm.write(f"Multi-config training: adaptive learning rate starting at "
                       f"{format_learning_rate(self.ladder.centre)}, "
                       f"one segment = {self.schedule.describe()}")
            tqdm.write("Each round compares the current learning rate with its neighbours and "
                       "follows the winner.")
        else:
            tqdm.write(f"Multi-config training: {len(self.candidates)} config sets, "
                       f"one segment = {self.schedule.describe()}")
        if self.base_config.multi_config_end_early:
            tqdm.write(f"End early is on: training stops after {self.end_early_patience} rounds in a "
                       f"row without a lower validation loss.")
        if self.base_config.multi_config_prune and self.ladder is None:
            if self.base_config.multi_config_prune_strategy == MultiConfigPruneStrategy.WORST_AVERAGE:
                rule = "the config with the worst average validation loss is dropped"
            else:
                rule = "configs that won nothing are dropped"
            tqdm.write(f"Pruning is on: every {self.base_config.multi_config_prune_after} round(s), "
                       f"{rule}.")
        if self.independent_lineages:
            tqdm.write("These configs train different sets of weights, so each one continues its own")
            tqdm.write("training instead of continuing from the round winner. Each round still reports")
            tqdm.write("and saves the best config so far.")
        for candidate in self.candidates:
            tqdm.write(f"  {candidate.label}: lr={candidate.config.learning_rate} "
                       f"optimizer={candidate.config.optimizer.optimizer} "
                       f"scheduler={candidate.config.learning_rate_scheduler}")
            for ignored in candidate.ignored_fields:
                tqdm.write(f"      ignored (taken from the base config) {ignored}")
        tqdm.write("=" * 78)
        tqdm.write("")

        self._write_json(os.path.join(self.multi_config_dir, "candidates.json"), report)

    @staticmethod
    def _write_json(path: str, data):
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
        except Exception:
            traceback.print_exc()
            tqdm.write(f"Could not write {path}")

    # --- training state snapshots --------------------------------------------------------------

    def _capture_state(self) -> dict:
        optimizer_state = None
        if self.model.optimizer is not None:
            optimizer_state = _detach_to_cpu(self.model.optimizer.state_dict())
            # create_optimizer() remaps groups by name when these keys are present, which is what lets
            # a candidate with different per-part learning rates reuse the state
            param_group_mapping = list(self.model.param_group_mapping or [])
            optimizer_state["param_group_mapping"] = param_group_mapping
            optimizer_state["param_group_optimizer_mapping"] = \
                [str(self.config.optimizer.optimizer) for _ in param_group_mapping]

        train_progress = self.model.train_progress

        return {
            'parameters': [p.detach().to(device="cpu", copy=True) for p in self.parameters],
            'optimizer': optimizer_state,
            'optimizer_name': str(self.config.optimizer.optimizer),
            'ema': _detach_to_cpu(self.model.ema.state_dict()) if self.model.ema is not None else None,
            'train_progress': {
                'epoch': train_progress.epoch,
                'epoch_step': train_progress.epoch_step,
                'epoch_sample': train_progress.epoch_sample,
                'global_step': train_progress.global_step,
            },
        }

    def _state_path(self, key: str) -> str:
        return os.path.join(self.state_dir, f"{key}.pt")

    def _store_state(self, key: str, state: dict):
        if self.base_config.multi_config_state_storage == MultiConfigStateStorage.RAM:
            self._ram_states[key] = state
        else:
            torch.save(state, self._state_path(key))

    def _load_state(self, key: str) -> dict:
        """Always returns an independent copy.

        The optimizer and EMA loaders take ownership of what they are handed, so a shared object would
        be mutated by the first candidate that used it and be wrong for the next one.
        """
        if self.base_config.multi_config_state_storage == MultiConfigStateStorage.RAM:
            state = self._ram_states[key]
            return {
                'parameters': state['parameters'],  # only ever read via copy_, never mutated
                'optimizer': _detach_to_cpu(state['optimizer']) if state['optimizer'] is not None else None,
                'optimizer_name': state['optimizer_name'],
                'ema': _detach_to_cpu(state['ema']) if state['ema'] is not None else None,
                'train_progress': dict(state['train_progress']),
            }

        return torch.load(self._state_path(key), weights_only=False, map_location="cpu")

    def _clear_state_dir(self):
        if not os.path.isdir(self.state_dir):
            return
        for filename in os.listdir(self.state_dir):
            if filename.endswith(".pt"):
                with contextlib.suppress(OSError):
                    os.remove(os.path.join(self.state_dir, filename))

    def _restore_state(self, key: str):
        state = self._load_state(key)

        with torch.no_grad():
            for parameter, saved in zip(self.parameters, state['parameters'], strict=True):
                parameter.copy_(saved.to(device=parameter.device, dtype=parameter.dtype))

        progress = state['train_progress']
        train_progress = self.model.train_progress
        train_progress.epoch = progress['epoch']
        train_progress.epoch_step = progress['epoch_step']
        train_progress.epoch_sample = progress['epoch_sample']
        train_progress.global_step = progress['global_step']

        self._pending_optimizer_state = state['optimizer']
        self._pending_optimizer_name = state['optimizer_name']
        self._pending_ema_state = state['ema']

    # --- candidate switching -------------------------------------------------------------------

    def _activate_config(self, config: TrainConfig):
        """Point the trainer at a candidate's settings and rebuild everything derived from them."""
        self.config = config
        self.model.train_config = config

        optimizer_state = self._pending_optimizer_state
        if optimizer_state is not None and self._pending_optimizer_name != str(config.optimizer.optimizer):
            # Adam moments mean nothing to Lion. Starting this candidate with a fresh optimizer is the
            # only correct option, at the cost of a warm-up disadvantage worth pointing out.
            tqdm.write(
                f"Optimizer changed from {self._pending_optimizer_name} to {config.optimizer.optimizer}; "
                f"this candidate starts with fresh optimizer state."
            )
            optimizer_state = None

        self.model.optimizer_state_dict = optimizer_state
        self.model.ema_state_dict = self._pending_ema_state

        # Drop the old optimizer and EMA before the replacements are built. Both allocate a full copy
        # of the trainable parameters on the train device, so holding two at once doubles that cost for
        # no reason and can be the difference between fitting and not.
        self.model.optimizer = None
        self.model.ema = None
        torch_gc()

        # rebuilds the optimizer (picking up the new learning rates and optimizer settings) and the
        # EMA wrapper, then clears the state dicts off the model
        init_model_parameters(self.model, self.model.parameters, self.train_device)

    def _rebuild_model_for(self, candidate: CandidateConfig):
        """Rebuild the adapter so it matches this candidate's set of trained weights.

        Only used by tournaments whose configs train different weights. The existing adapter is
        unhooked first, otherwise both the old and the new one would apply to the model. ``setup_model``
        creates a fresh adapter and a fresh optimizer for it; the candidate's own saved state, if it
        has any, is restored on top afterwards.
        """
        self.config = candidate.config
        self.model.train_config = candidate.config

        for adapter in self.model.adapters():
            adapter.remove_hook_from_module()

        self.model.lora_state_dict = None
        self.model.optimizer_state_dict = None
        self.model.ema_state_dict = None
        self.model.optimizer = None
        self.model.ema = None
        torch_gc()

        # a fixed seed keeps a fresh adapter reproducible from run to run
        torch.manual_seed(ADAPTER_INIT_SEED)

        self.model_setup.setup_model(self.model, candidate.config)
        self.parameters = self.model.parameters.parameters()
        self.model_setup.setup_train_device(self.model, candidate.config)

    def _rebuild_train_data_loader(self):
        """Reposition the training data at the start of the segment.

        mgds only honours a mid-epoch start offset at construction time, and each call to
        ``start_next_epoch`` advances the shuffle. Rebuilding the pipeline from the restored progress
        is therefore what makes every candidate in a round see the same batches in the same order.

        Turning off deterministic data skips the rebuild on epoch-aligned boundaries, where reusing the
        pipeline merely means each candidate trains on a different shuffle of the same data. A
        mid-epoch boundary is rebuilt regardless: reusing the pipeline there would put the data and the
        recorded progress out of step.
        """
        mid_epoch = self.model.train_progress.epoch_step > 0
        if not self.base_config.multi_config_deterministic_data and not mid_epoch:
            return

        old_data_loader = self.data_loader
        self.data_loader = self.create_data_loader(
            self.model, self.model_setup, self.model.train_progress
        )
        _release_data_loader(old_data_loader)

    def _rebuild_validation_data_loader(self):
        """Rebuild the validation pipeline pinned to epoch 0.

        Validation modules vary their augmentation with the epoch counter, and the built-in behaviour
        advances that counter on every validation pass. Pinning it keeps the validation set identical
        between candidates (so a round is a fair comparison) and across rounds (so the curve in
        tensorboard measures the model rather than the data).
        """
        if not self.base_config.multi_config_deterministic_data:
            return

        old_data_loader = self.validation_data_loader
        self.validation_data_loader = self.create_data_loader(
            self.model, self.model_setup, TrainProgress(), is_validation=True
        )
        _release_data_loader(old_data_loader)

    # --- the tournament ------------------------------------------------------------------------

    def train(self):
        train_progress = self.model.train_progress

        # measure the sampling interval from wherever this run starts, so resuming a backup does not
        # sample immediately just because the counters are already large
        self._last_sample_epoch = train_progress.epoch
        self._last_sample_step = train_progress.global_step

        if not self.independent_lineages:
            # the state every candidate of the first round starts from. Independent lineages have no
            # shared state to store: each candidate builds its own on its first segment.
            self._store_state(BASE_STATE_KEY, self._capture_state())

        while True:
            if self.commands.get_stop_command():
                self.finished_reason = "stopped by user"
                break

            if train_progress.epoch >= self.config.epochs:
                self.finished_reason = "reached the configured epoch count"
                break

            round_complete = self._run_round()

            if self.commands.get_stop_command():
                self.finished_reason = "stopped by user"
                break

            if not round_complete:
                self.finished_reason = self.finished_reason or "training finished"
                break

        self.segment_end_check = None
        self.suppress_scheduled_validation = False
        self.suppress_scheduled_actions = False

    def _run_round(self) -> bool:
        """Run one round over every candidate and promote the winner.

        Returns True when the round ended on a segment boundary and another round should follow, and
        False when training reached its end during this round.
        """
        train_progress = self.model.train_progress
        round_start_epoch = train_progress.epoch
        round_start_step = train_progress.global_step
        round_start = TrainProgress(
            epoch=train_progress.epoch,
            epoch_step=train_progress.epoch_step,
            epoch_sample=train_progress.epoch_sample,
            global_step=train_progress.global_step,
        )

        self.schedule.unlock()
        results: list[CandidateRoundResult] = []
        reached_end_of_training = False

        # the adaptive mode rebuilds its candidates every round, around whichever learning rate is
        # currently winning
        if self.ladder is not None:
            self.candidates = build_ladder_candidates(self.base_config, self.ladder)
            tqdm.write(f"[multi-config] round {self.round_index + 1} learning rates: "
                       f"{self.ladder.describe()}")

        for candidate in self.candidates:
            if self.commands.get_stop_command():
                # discard the interrupted candidate: it trained a different number of steps than the
                # ones before it, so its validation loss is not comparable
                break

            self.callbacks.on_update_status(
                f"multi-config round {self.round_index + 1}: {candidate.label}"
            )
            tqdm.write(
                f"[multi-config] round {self.round_index + 1}, {candidate.label}: "
                f"training {self.schedule.describe()} from epoch {round_start_epoch} "
                f"(step {round_start_step})"
            )

            result = self._run_candidate_segment(candidate, round_start)
            results.append(result)

            if self.commands.get_stop_command() and not result.segment_completed:
                results.pop()
                break

            if not result.segment_completed:
                reached_end_of_training = True

            if len(results) == 1:
                # Freeze the round to the length of the first candidate's segment. For step- and
                # epoch-based intervals this is already what the others would do; for wall-clock
                # intervals it is what stops a slower candidate from being judged after fewer steps.
                self.schedule.lock_to(self.model.train_progress)

        if not results:
            # Nothing in this round was judged, so the model is sitting on a partial segment of an
            # arbitrary candidate. Rewind to the last promoted state, which is the last one that
            # actually won a comparison, so the final save is a state the tournament stands behind.
            tqdm.write(
                "[multi-config] stopped before any config finished a segment; "
                "rewinding to the last promoted state"
            )
            self._rewind_to_promoted()
            self.finished_reason = self.finished_reason or "stopped before any config finished a segment"
            return False

        if all(result.steps_trained == 0 for result in results):
            # The epoch counter can sit at the end of an epoch that has no batches left when a
            # step-based interval finishes exactly on an epoch boundary. Reaching this on the final
            # epoch means there is genuinely nothing left to train, not that something went wrong.
            tqdm.write("[multi-config] no training steps left; finishing")
            self._activate_config(self._promoted_config)
            self.finished_reason = "training finished"
            return False

        winner_index = pick_winner([result.score for result in results])

        if winner_index is None:
            trained = sum(result.steps_trained for result in results)
            raise RuntimeError(
                "Multi-config training could not compare configs: none of them produced a validation "
                f"loss after {trained} training step(s) in round {self.round_index + 1}. Either mark "
                "a concept as a validation concept, or set 'Auto Validation Split %' on the general "
                "tab to hold part of every concept back, and check that what it selects contains "
                "usable samples."
            )

        winner = results[winner_index]
        self._promote(winner, results, round_start_epoch, round_start_step)
        self._prune_candidates(results)

        self.round_index += 1

        if self._should_end_early():
            self.finished_reason = (
                f"stopped early: {self.rounds_without_improvement} rounds in a row without "
                f"improving on a validation loss of {self.best_score:.6f} from round {self.best_round}"
            )
            tqdm.write(f"[multi-config] {self.finished_reason}")
            return False

        train_progress = self.model.train_progress
        if train_progress.global_step == round_start_step and train_progress.epoch == round_start_epoch:
            # A round that advances neither steps nor epochs would repeat forever. This should be
            # unreachable, but the failure mode is a hang, so it is worth refusing explicitly.
            self.finished_reason = "a round made no progress; stopping to avoid an endless loop"
            tqdm.write(f"[multi-config] {self.finished_reason}")
            return False

        return not reached_end_of_training

    def _run_candidate_segment(
            self,
            candidate: CandidateConfig,
            round_start: TrainProgress,
    ) -> CandidateRoundResult:
        self._begin_candidate(candidate, round_start)
        self._rebuild_train_data_loader()

        train_progress = self.model.train_progress
        start_step = train_progress.global_step

        self.schedule.begin(train_progress, time.monotonic())
        self.segment_end_check = lambda progress: self.schedule.is_complete(progress, time.monotonic())
        self.suppress_scheduled_validation = True
        self.suppress_scheduled_actions = True

        try:
            super().train()
        finally:
            self.segment_end_check = None

        segment_completed = self.segment_ended
        steps_trained = train_progress.global_step - start_step
        self._normalise_to_locked_target(train_progress)

        validation_result = None
        error = None
        if steps_trained > 0 and not (self.commands.get_stop_command() and not segment_completed):
            try:
                self._rebuild_validation_data_loader()
                validation_result = self.run_validation(
                    train_progress,
                    tensorboard_prefix=f"loss/multi_config/{candidate.slug}",
                )
            except Exception as e:
                traceback.print_exc()
                error = str(e)

        score = score_validation_result(validation_result, self.base_config.multi_config_selection)

        self._store_state(candidate.slug, self._capture_state())
        torch_gc()

        tqdm.write(
            f"[multi-config] {candidate.label}: {steps_trained} step(s), "
            f"score={'n/a' if score is None else f'{score:.6f}'}"
            f"{'' if segment_completed else ' (training ended during this segment)'}"
        )

        return CandidateRoundResult(
            candidate=candidate,
            score=score,
            validation=validation_result,
            steps_trained=steps_trained,
            segment_completed=segment_completed,
            error=error,
        )

    def _begin_candidate(self, candidate: CandidateConfig, round_start: TrainProgress):
        """Put the model into the state this candidate's segment should start from."""
        if not self.independent_lineages:
            # every candidate starts the round from the state the previous round promoted
            self._restore_state(BASE_STATE_KEY)
            self._activate_config(candidate.config)
            return

        # This candidate trains its own set of weights, so it continues its own lineage. The adapter
        # has to be rebuilt because the candidate that ran before it left a different one in place.
        self._rebuild_model_for(candidate)

        if self._has_state(candidate.slug):
            self._restore_state(candidate.slug)
            self._activate_config(candidate.config)
        else:
            # first segment for this candidate: keep the adapter setup_model just built, and start
            # from where the round starts rather than from wherever the previous candidate finished
            self._set_progress(round_start)

    def _rewind_to_promoted(self):
        """Put the model back on the last state that actually won a comparison.

        Used when a round is interrupted before anything was judged, so the model left behind is one
        the tournament stands behind rather than a partial segment of an arbitrary candidate.
        """
        if self.independent_lineages:
            if self._promoted_candidate is None or not self._has_state(self._promoted_candidate.slug):
                # nothing has been promoted yet, so the first candidate's partial segment is all there
                # is - keep it rather than throwing the only training away
                return
            self._rebuild_model_for(self._promoted_candidate)
            self._restore_state(self._promoted_candidate.slug)
        else:
            self._restore_state(BASE_STATE_KEY)

        self._activate_config(self._promoted_config)

    def _set_progress(self, source: TrainProgress):
        train_progress = self.model.train_progress
        train_progress.epoch = source.epoch
        train_progress.epoch_step = source.epoch_step
        train_progress.epoch_sample = source.epoch_sample
        train_progress.global_step = source.global_step

    def _has_state(self, key: str) -> bool:
        if self.base_config.multi_config_state_storage == MultiConfigStateStorage.RAM:
            return key in self._ram_states
        return os.path.isfile(self._state_path(key))

    def _normalise_to_locked_target(self, train_progress: TrainProgress):
        """Adopt the first candidate's epoch bookkeeping when a locked round lands on the same step.

        "The last step of epoch 3" and "the start of epoch 4" are the same position but different
        counters, and which one a segment reports depends on whether it stopped inside the batch loop
        or at the epoch rollover. Only the second form is canonical - the first would make the next
        round replay an epoch that has no batches left. This only applies to wall-clock intervals,
        the only ones that lock.
        """
        target = self.schedule.locked_target
        if target is None or train_progress.global_step != target.global_step:
            return
        train_progress.epoch = target.epoch
        train_progress.epoch_step = target.epoch_step
        train_progress.epoch_sample = target.epoch_sample

    def _record_best(self, winner: CandidateRoundResult):
        """Keep the state behind the lowest validation score of the run, and count stalled rounds.

        The winner is the best of its round, so comparing it to the running best is the same as
        asking whether any config in the round improved on anything seen so far.
        """
        if winner.score is None:
            self.rounds_without_improvement += 1
            return

        if self.best_score is not None and winner.score >= self.best_score:
            self.rounds_without_improvement += 1
            tqdm.write(
                f"[multi-config] no improvement on {self.best_score:.6f} from round "
                f"{self.best_round} ({self.rounds_without_improvement} round(s) in a row)"
            )
            return

        self.best_score = winner.score
        self.best_round = self.round_index + 1
        self.best_candidate = winner.candidate
        self.rounds_without_improvement = 0

        # the model is already sitting on the winner's state at this point
        self._store_state(BEST_STATE_KEY, self._capture_state())

    @property
    def end_early_patience(self) -> int:
        configured = int(self.base_config.multi_config_end_early_rounds or 0)
        return configured if configured >= 1 else DEFAULT_END_EARLY_PATIENCE

    def _should_end_early(self) -> bool:
        return (
            self.base_config.multi_config_end_early
            and self.rounds_without_improvement >= self.end_early_patience
        )

    def _prune_candidates(self, results: list[CandidateRoundResult]):
        """Narrow the field at the end of each pruning window.

        A config that keeps losing costs a full segment of training per round for a result the run
        will not use. Once only one is left the tournament is effectively ordinary training, and it
        carries on that way until the epochs run out or End Early stops it.
        """
        if not self.base_config.multi_config_prune or self.ladder is not None:
            return
        if len(self.candidates) <= 1:
            return

        window = int(self.base_config.multi_config_prune_after or 0)
        if window < 1:
            return

        self._rounds_since_prune += 1
        for result in results:
            self._window_wins.setdefault(result.candidate.name, 0)
            if result.score is not None:
                self._candidate_scores.setdefault(result.candidate.name, []).append(result.score)
        self._window_wins[self._promoted_candidate.name] = \
            self._window_wins.get(self._promoted_candidate.name, 0) + 1

        if self._rounds_since_prune < window:
            return

        if self.base_config.multi_config_prune_strategy == MultiConfigPruneStrategy.WORST_AVERAGE:
            survivors, reason = self._survivors_by_average()
        else:
            survivors, reason = self._survivors_by_wins(window)

        dropped = [c.name for c in self.candidates if c not in survivors]
        if dropped:
            tqdm.write(
                f"[multi-config] dropping {', '.join(dropped)} {reason} | continuing with "
                f"{', '.join(c.name for c in survivors)}"
            )
            self.candidates = survivors

        self._window_wins = {}
        self._rounds_since_prune = 0

    def _survivors_by_wins(self, window: int) -> tuple[list[CandidateConfig], str]:
        survivors = [c for c in self.candidates if self._window_wins.get(c.name, 0) > 0]

        if not survivors:
            # can only happen if every round in the window failed to produce a winner
            survivors = self.candidates

        return survivors, f"after {window} round(s) without a win"

    def _survivors_by_average(self) -> tuple[list[CandidateConfig], str]:
        """Drop the one config with the worst average validation loss over the whole run.

        Ranks on how far behind a config actually is rather than on whether it happened to come
        first, so a config that is consistently a close second survives while one that is never in
        contention goes. The average covers every round the config has run, not just the ones since
        the last check - a rate that was hopeless for the first twenty rounds should not be rescued
        by one good round right before a check.

        Only one is dropped per check, because after removing the worst the remaining averages say
        nothing new until they have been measured against each other again.
        """
        averages = {}
        rounds = {}
        for candidate in self.candidates:
            scores = self._candidate_scores.get(candidate.name)
            rounds[candidate.name] = len(scores) if scores else 0
            # a config that never produced a usable validation loss is worse than any that did
            averages[candidate.name] = sum(scores) / len(scores) if scores else math.inf

        worst = max(self.candidates, key=lambda c: (averages[c.name], c.index))
        survivors = [c for c in self.candidates if c is not worst]

        average = averages[worst.name]
        if math.isinf(average):
            shown = "no validation loss"
        else:
            shown = f"average loss {average:.6f} over {rounds[worst.name]} round(s)"
        return survivors, f"with the worst {shown}"

    def _promote(
            self,
            winner: CandidateRoundResult,
            results: list[CandidateRoundResult],
            round_start_epoch: int,
            round_start_step: int,
    ):
        """Leave the model on the round's winner, and save it."""
        if self.independent_lineages:
            # no state changes hands; the winner just becomes the model that gets saved and sampled,
            # and the one a stop or the end of training leaves behind
            self._rebuild_model_for(winner.candidate)
            self._restore_state(winner.candidate.slug)
            self._activate_config(winner.candidate.config)
            self._promoted_config = winner.candidate.config
        else:
            self._restore_state(winner.candidate.slug)
            self._activate_config(winner.candidate.config)
            self._promoted_config = winner.candidate.config
            self._store_state(BASE_STATE_KEY, self._capture_state())

        self._promoted_candidate = winner.candidate

        if self.ladder is not None and winner.candidate.config.learning_rate is not None:
            moved = self.ladder.advance(winner.candidate.config.learning_rate)
            if not moved:
                tqdm.write(f"[multi-config] learning rate stays at "
                           f"{format_learning_rate(self.ladder.centre)}")

        self._record_best(winner)

        train_progress = self.model.train_progress

        # Re-log the winner under the normal validation series so the tensorboard curve everyone
        # already reads stays continuous across rounds.
        if winner.validation is not None:
            for label, loss in winner.validation.per_concept.items():
                self.tensorboard.add_scalar(
                    f"loss/validation_step/{label}", loss, train_progress.global_step
                )
            if winner.validation.total_average is not None and len(winner.validation.per_concept) > 1:
                self.tensorboard.add_scalar(
                    "loss/validation_step/total_average",
                    winner.validation.total_average,
                    train_progress.global_step,
                )
        if winner.score is not None:
            self.tensorboard.add_scalar(
                "loss/multi_config/winner", winner.score, train_progress.global_step
            )

        ranking = ", ".join(
            f"{result.candidate.name}={'n/a' if result.score is None else f'{result.score:.6f}'}"
            for result in results
        )
        tqdm.write(
            f"[multi-config] round {self.round_index + 1} winner: {winner.candidate.label} "
            f"(score {'n/a' if winner.score is None else f'{winner.score:.6f}'}) | {ranking}"
        )
        self.callbacks.on_update_status(
            f"multi-config round {self.round_index + 1}: {winner.candidate.name} won"
        )

        self.history.append({
            'round': self.round_index + 1,
            'start_epoch': round_start_epoch,
            'start_global_step': round_start_step,
            'end_epoch': train_progress.epoch,
            'end_global_step': train_progress.global_step,
            'selection': str(self.base_config.multi_config_selection),
            'winner': winner.candidate.name,
            'results': [result.to_dict() for result in results],
        })
        self._write_json(os.path.join(self.multi_config_dir, "results.json"), self.history)

        if self._should_sample_after_round(train_progress):
            self._last_sample_epoch = train_progress.epoch
            self._last_sample_step = train_progress.global_step
            try:
                self.run_sampling(train_progress)
            except Exception:
                traceback.print_exc()
                tqdm.write("Error during sampling, proceeding without sampling")

        if self.base_config.multi_config_save_each_round:
            self._save_round_model(winner)

    def _should_sample_after_round(self, train_progress: TrainProgress) -> bool:
        """Apply the user's sampling interval to promoted models only.

        The interval is measured between promotions rather than evaluated against the raw counters,
        because a round rarely lands on an exact multiple of the interval - an "every 200 steps"
        setting with a 150-step validation interval would otherwise never fire.
        """
        unit = self.config.sample_after_unit
        interval = self.config.sample_after

        if unit == TimeUnit.NEVER:
            return False
        if not self.single_action_elapsed(
                "multi_config_sample_skip_first", self.config.sample_skip_first, unit, train_progress):
            return False
        if unit == TimeUnit.ALWAYS:
            return True
        if interval <= 0:
            return False
        if unit.is_time_unit():
            # already measured against wall-clock elapsed time, so the counters don't come into it
            return self.repeating_action_needed(
                "multi_config_sample", interval, unit, train_progress, start_at_zero=False
            )
        if unit == TimeUnit.EPOCH:
            return train_progress.epoch - self._last_sample_epoch >= int(interval)
        return train_progress.global_step - self._last_sample_step >= int(interval)

    def _save_round_model(self, winner: CandidateRoundResult):
        """Save the winning model for this round.

        Reuses the normal save path (which handles EMA and schedule-free optimizers correctly) and
        only borrows the filename prefix to record which candidate produced the file.
        """
        original_prefix = self.config.save_filename_prefix
        self.config.save_filename_prefix = \
            f"{original_prefix}round{self.round_index + 1:03d}-{winner.candidate.slug}-"
        try:
            self.model.to(self.temp_device)
            self.save_model_snapshot(self.model.train_progress, True)
        finally:
            self.config.save_filename_prefix = original_prefix
            self.model_setup.setup_train_device(self.model, self.config)

    # --- teardown ------------------------------------------------------------------------------

    def _restore_best_state(self):
        """Put the lowest-scoring state of the run back on the model, ready for the final save.

        The run ends wherever the last round left it, which is not necessarily the best it ever was -
        the loss may have climbed again, or the run may have been stopped. Restoring here means the
        model written to the output destination is always the best one the tournament saw.
        """
        if self.best_candidate is None or not self._has_state(BEST_STATE_KEY):
            return

        if self.best_round == self.round_index and self._promoted_candidate is self.best_candidate:
            # the last round was the best one, so the model already holds it
            return

        tqdm.write(
            f"[multi-config] restoring the best model: {self.best_candidate.name} from round "
            f"{self.best_round}, validation loss {self.best_score:.6f}"
        )
        self.callbacks.on_update_status("multi-config: restoring the best model")

        try:
            if self.independent_lineages:
                self._rebuild_model_for(self.best_candidate)
            self._restore_state(BEST_STATE_KEY)
            self._activate_config(self.best_candidate.config)
        except Exception:
            traceback.print_exc()
            tqdm.write("Could not restore the best model; saving the final state instead")

    def end(self):
        self._restore_best_state()

        if self.candidates and self.history:
            summary = {
                'rounds': len(self.history),
                'finished_reason': self.finished_reason,
                'selection': str(self.base_config.multi_config_selection),
                'best_score': self.best_score,
                'best_round': self.best_round,
                'best_candidate': self.best_candidate.name if self.best_candidate else None,
                'wins': self._win_counts(),
                'history': self.history,
            }
            if self.ladder is not None:
                summary['final_learning_rate'] = self.ladder.centre
            self._write_json(os.path.join(self.multi_config_dir, "summary.json"), summary)

            tqdm.write("")
            tqdm.write("=" * 78)
            tqdm.write(f"Multi-config training finished after {len(self.history)} round(s) "
                       f"({self.finished_reason}).")
            for name, wins in sorted(self._win_counts().items(), key=lambda item: -item[1]):
                tqdm.write(f"  {name}: {wins} round(s) won")
            if self.best_candidate is not None:
                tqdm.write(f"Best validation loss {self.best_score:.6f} from round {self.best_round} "
                           f"({self.best_candidate.name}); that is the model being saved.")
            if self.ladder is not None:
                tqdm.write(f"Learning rate finished at {format_learning_rate(self.ladder.centre)}.")
            tqdm.write(f"Details: {os.path.join(self.multi_config_dir, 'summary.json')}")
            tqdm.write("=" * 78)
            tqdm.write("")

        self._ram_states.clear()

        super().end()

    def _win_counts(self) -> dict[str, int]:
        # built from the history rather than the candidate list, because the adaptive mode's
        # candidates change every round and self.candidates only holds the most recent set
        counts: dict[str, int] = {}
        if self.ladder is None:
            counts = {candidate.name: 0 for candidate in self.candidates}
        for entry in self.history:
            counts[entry['winner']] = counts.get(entry['winner'], 0) + 1
        return counts
