"""Tests for the Automagic3 optimizer and its integration into OneTrainer.

Automagic3 sets its own learning rate from a pooled vote over per-element update sign histories, so
the things worth pinning down are that it actually optimises, that the controller moves the rate in
the right direction, and that the OneTrainer-specific adaptations hold: the fused back pass path
applies the vote, and the state survives a save/load round trip.

Run with:  pytest tests/test_automagic3.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

torch = pytest.importorskip("torch")

from modules.util import create  # noqa: E402
from modules.util.config.TrainConfig import TrainConfig  # noqa: E402
from modules.util.enum.Optimizer import Optimizer  # noqa: E402
from modules.util.NamedParameterGroup import (  # noqa: E402
    NamedParameterGroup,
    NamedParameterGroupCollection,
)
from modules.util.optimizer.automagic3 import Automagic3  # noqa: E402
from modules.util.optimizer_util import OPTIMIZER_DEFAULT_PARAMETERS, change_optimizer  # noqa: E402


def quadratic(size: int = 64, start: float = 1.0) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.full((size,), start))


def run_steps(optimizer, param, steps: int, target: float = 0.0, fused: bool = False):
    for _ in range(steps):
        loss = ((param - target) ** 2).mean()
        loss.backward()
        if fused:
            # what the trainer does when Fused Back Pass is on: per-parameter steps, no step() call
            optimizer.step_parameter(param, optimizer.param_groups[0], 0)
            optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return param


# --- the optimizer itself -------------------------------------------------------------------------


def test_it_moves_the_parameter_towards_the_target():
    param = quadratic()
    optimizer = Automagic3([param], lr=1e-3, polarity_history=4)

    before = param.detach().clone()
    run_steps(optimizer, param, 30)

    assert param.abs().mean() < before.abs().mean(), "the parameter should have moved downhill"
    assert torch.isfinite(param).all()


def test_the_learning_rate_climbs_while_the_direction_holds():
    # every step pushes the same way, so the sign windows fill with agreement and the pooled vote
    # says "step too small"
    param = quadratic(size=4096, start=50.0)
    optimizer = Automagic3([param], lr=1e-6, polarity_history=4)

    start_lr = optimizer.get_learning_rates()[0]
    run_steps(optimizer, param, 40)
    end_lr = optimizer.get_learning_rates()[0]

    assert end_lr > start_lr, "a consistent descent direction should raise the rate"


def test_the_rails_bound_the_adapted_rate():
    param = quadratic(size=4096, start=50.0)
    optimizer = Automagic3([param], lr=1e-6, min_lr=5e-7, max_lr=2e-6, polarity_history=4)

    run_steps(optimizer, param, 60)

    for lr in optimizer.get_learning_rates():
        assert 5e-7 <= lr <= 2e-6, f"{lr} escaped the configured rails"


def test_min_lr_above_max_lr_is_rejected():
    with pytest.raises(ValueError):
        Automagic3([quadratic()], lr=1e-6, min_lr=1e-3, max_lr=1e-6)


def test_the_starting_rate_is_clamped_into_the_rails():
    param = quadratic()
    optimizer = Automagic3([param], lr=1e-2, min_lr=1e-8, max_lr=1e-4)
    run_steps(optimizer, param, 1)

    assert optimizer.get_learning_rates()[0] <= 1e-4


@pytest.mark.parametrize("history", [2, 4, 8, 16])
def test_it_works_across_history_lengths(history):
    param = quadratic(size=512, start=10.0)
    optimizer = Automagic3([param], lr=1e-3, polarity_history=history)

    run_steps(optimizer, param, 2 * history + 5)

    assert torch.isfinite(param).all()
    assert param.abs().mean() < 10.0


def test_the_history_length_is_clamped_to_what_the_packing_supports():
    assert Automagic3([quadratic()], polarity_history=1).param_groups[0]["polarity_history"] == 2
    assert Automagic3([quadratic()], polarity_history=999).param_groups[0]["polarity_history"] == 64


def test_one_rate_is_shared_by_the_whole_group():
    # the point of v3 over per-parameter rates: coupled tensors cannot fight each other
    a, b = quadratic(size=128), quadratic(size=128)
    optimizer = Automagic3([a, b], lr=1e-3, polarity_history=4)

    for _ in range(20):
        (((a - 0.0) ** 2).mean() + ((b - 0.0) ** 2).mean()).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    assert len(optimizer.get_learning_rates()) == 1, "one rate per param group, not per tensor"


def test_separate_param_groups_get_separate_rates():
    a, b = quadratic(size=128), quadratic(size=128)
    optimizer = Automagic3([{"params": [a]}, {"params": [b]}], lr=1e-3, polarity_history=4)

    assert len(optimizer.get_learning_rates()) == 2


def test_the_average_rate_is_reported():
    optimizer = Automagic3([{"params": [quadratic()]}, {"params": [quadratic()]}], lr=1e-3)
    assert optimizer.get_avg_learning_rate() == pytest.approx(1e-3)


# --- the OneTrainer adaptations ---------------------------------------------------------------------


def test_the_fused_back_pass_path_optimises_and_still_votes():
    """With a fused back pass the trainer never calls step(), so the vote is flushed in zero_grad."""
    param = quadratic(size=4096, start=50.0)
    optimizer = Automagic3([param], lr=1e-6, polarity_history=4)

    start_lr = optimizer.get_learning_rates()[0]
    run_steps(optimizer, param, 40, fused=True)

    assert param.abs().mean() < 50.0, "step_parameter should have updated the weights"
    assert optimizer.get_learning_rates()[0] > start_lr, "the pooled vote must still be applied"


def test_fused_and_unfused_paths_agree():
    # the optimizer draws from the global RNG for stochastic rounding, so both runs have to start
    # from the same RNG state to be comparable
    torch.manual_seed(0)
    fused_param = quadratic(size=256, start=5.0)
    run_steps(Automagic3([fused_param], lr=1e-4, max_lr=1e-3, polarity_history=4),
              fused_param, 25, fused=True)

    torch.manual_seed(0)
    plain_param = quadratic(size=256, start=5.0)
    run_steps(Automagic3([plain_param], lr=1e-4, max_lr=1e-3, polarity_history=4),
              plain_param, 25, fused=False)

    assert torch.allclose(fused_param, plain_param, rtol=1e-5, atol=1e-7)


def test_the_adapted_learning_rate_survives_a_save_and_load():
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.randn(256))
    optimizer = Automagic3([param], lr=1e-4, max_lr=1e-3, polarity_history=4)
    run_steps(optimizer, param, 12)

    adapted_lr = optimizer.get_learning_rates()[0]
    assert adapted_lr != pytest.approx(1e-4), "the controller should have moved the rate by now"

    resumed_param = torch.nn.Parameter(param.detach().clone())
    resumed = Automagic3([resumed_param], lr=1e-4, max_lr=1e-3, polarity_history=4)
    resumed.load_state_dict(optimizer.state_dict())

    # resuming must not throw away what the controller learned and restart from the configured rate
    assert resumed.get_learning_rates()[0] == pytest.approx(adapted_lr, rel=1e-6)


def test_the_whole_optimizer_state_comes_back():
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.randn(256))
    optimizer = Automagic3([param], lr=1e-4, max_lr=1e-3, polarity_history=4)
    run_steps(optimizer, param, 12)

    resumed_param = torch.nn.Parameter(param.detach().clone())
    resumed = Automagic3([resumed_param], lr=1e-4, max_lr=1e-3, polarity_history=4)
    resumed.load_state_dict(optimizer.state_dict())

    before, after = optimizer.state[param], resumed.state[resumed_param]
    assert sorted(before) == sorted(after)
    for key, value in before.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value.float(), after[key].float()), f"{key} did not survive the resume"
        else:
            assert value == after[key], f"{key} did not survive the resume"

    # and it keeps training from there
    run_steps(resumed, resumed_param, 5)
    assert torch.isfinite(resumed_param).all()


def test_a_resume_with_a_different_history_length_starts_the_window_fresh():
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.randn(64))
    optimizer = Automagic3([param], lr=1e-4, polarity_history=4)
    run_steps(optimizer, param, 10)

    resumed_param = torch.nn.Parameter(param.detach().clone())
    resumed = Automagic3([resumed_param], lr=1e-4, polarity_history=8)
    resumed.load_state_dict(optimizer.state_dict())

    state = resumed.state[resumed_param]
    assert state["sign_history"].shape[0] == 8, "the window must match the new setting"
    assert state["hist_fill"] == 0, "a window of a different shape cannot be carried over"
    # the adapted rate is still kept, only the sign history restarts
    assert resumed.get_learning_rates()[0] == pytest.approx(optimizer.get_learning_rates()[0], rel=1e-6)


# --- integration ------------------------------------------------------------------------------------


def test_the_enum_advertises_what_the_trainer_needs():
    assert Optimizer.AUTOMAGIC3.is_adaptive, "the lr is reported from the optimizer, not the config"
    assert Optimizer.AUTOMAGIC3.supports_fused_back_pass()
    assert not Optimizer.AUTOMAGIC3.is_schedule_free


def test_the_reported_learning_rates_come_from_the_optimizer():
    a, b = quadratic(size=64), quadratic(size=64)
    optimizer = Automagic3([{"params": [a]}, {"params": [b]}], lr=1e-3)

    # the configured rates are ignored in favour of whatever the controller has settled on
    reported = Optimizer.AUTOMAGIC3.maybe_adjust_lrs({"group0": 9.9, "group1": 9.9}, optimizer)

    assert reported == {"group0": pytest.approx(1e-3), "group1": pytest.approx(1e-3)}


def test_it_has_defaults_registered():
    defaults = OPTIMIZER_DEFAULT_PARAMETERS[Optimizer.AUTOMAGIC3]
    assert defaults["polarity_history"] == 8
    assert defaults["min_lr"] == 1e-8
    assert defaults["max_lr"] == 1e3


def test_selecting_it_brings_its_own_parameters():
    config = TrainConfig.default_values()
    config.optimizer.optimizer = Optimizer.AUTOMAGIC3
    config.optimizer.from_dict(change_optimizer(config).to_dict())

    assert config.optimizer.polarity_history == 8
    assert config.optimizer.clip_threshold == 1.0
    assert config.optimizer.weight_decay == 0.0


def test_create_optimizer_builds_it_from_a_config():
    config = TrainConfig.default_values()
    config.learning_rate = 1e-5
    config.optimizer.optimizer = Optimizer.AUTOMAGIC3
    config.optimizer.from_dict(change_optimizer(config).to_dict())

    param = quadratic()
    collection = NamedParameterGroupCollection()
    collection.add_group(NamedParameterGroup(unique_name="w", parameters=[param], learning_rate=None))

    optimizer = create.create_optimizer(collection, None, config)

    assert isinstance(optimizer, Automagic3)
    assert optimizer.param_groups[0]["polarity_history"] == 8
    assert optimizer.get_learning_rates()[0] == pytest.approx(1e-5)

    run_steps(optimizer, param, 5)
    assert torch.isfinite(param).all()


def test_it_is_offered_as_a_multi_config_sweep_value():
    from modules.util.enum.MultiConfigSweepSetting import MultiConfigSweepSetting
    from modules.util.multi_config_sweep import sweep_options, validate_sweep_value

    config = TrainConfig.default_values()
    assert "AUTOMAGIC3" in sweep_options(config, MultiConfigSweepSetting.OPTIMIZER)
    assert validate_sweep_value(config, MultiConfigSweepSetting.OPTIMIZER, "AUTOMAGIC3") is None
