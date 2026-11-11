# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Thomas J. Pope, C. D. Rankine, B. Li
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.
#
# Citations:
#   Junkawitsch et al., "XANESNET: A Modular, Extensible, and Flexible Machine Learning Framework for Spectroscopy."
#   Rankine et al., "A Deep Neural Network for the Rapid Prediction of X-ray Absorption Spectra."
#   Rankine et al., "Accurate, affordable, and generalizable machine learning simulations of transition metal x-ray absorption spectra using the XANESNET deep neural network."
#   Penfold et al., "Field-Aware Energy-Conditioned Message Passing Neural Networks for Absorber-Centred Modelling of X-ray Spectroscopy."
#   Penfold et al., "A deep neural network for valence-to-core X-ray emission spectroscopy."
#   Falbo et al., "On the Analysis of X-ray Absorption Spectra for Polyoxometallates."
#   Madkhali et al., "Enhancing the Analysis of Disorder in X-ray Absorption Spectra: Application of Deep Neural Networks to T-Jump X-ray Probe Experiments."
#   Madkhali et al., "The Role of Structural Representation in the Performance of a Deep Neural Network for X-ray Spectroscopy."

"""Unit tests for ``Checkpointer``: state collection, periodic saving and member bookkeeping."""

from types import SimpleNamespace

import pytest
import torch

from xanesnet.checkpointing import Checkpointer
from xanesnet.checkpointing.checkpointer import state_to_cpu
from xanesnet.serialization.checkpoints import Checkpoint
from xanesnet.serialization.config import Config

###############################################################################
################################# TEST DOUBLES ################################
###############################################################################


class _Component:
    def __init__(self, signature: dict, state: dict) -> None:
        self.signature = Config(signature)
        self._state = state

    def state_dict(self) -> dict:
        return dict(self._state)


def make_trainer(seed: int = 0, epochs: int = 3, warmup: bool = False, lr: float = 0.1) -> SimpleNamespace:
    torch.manual_seed(seed)
    return SimpleNamespace(
        model=torch.nn.Linear(2, 2),
        completed_epochs=epochs,
        optimizer=_Component({"optimizer_type": "adam", "lr": lr}, {"step": torch.tensor(float(epochs))}),
        epoch_lr_scheduler=_Component({"lr_scheduler_type": "step"}, {"last_epoch": epochs}),
        warmup_lr_scheduler=_Component({"lr_scheduler_type": "linear"}, {"last_epoch": 1}) if warmup else None,
    )


@pytest.fixture
def signature() -> Config:
    return Config({"model": {"model_type": "mlp"}})


def make_checkpointer(signature, trainers, save_dir=None, save_interval=None) -> Checkpointer:
    return Checkpointer(save_dir, save_interval, signature, trainer_provider=lambda: trainers)


###############################################################################
################################## state_to_cpu ###############################
###############################################################################


def test_tensors_are_copied_so_snapshots_do_not_follow_in_place_updates():
    # On CPU ``.cpu()`` is a no-op, so only the explicit clone protects the snapshot.
    original = torch.ones(3)
    copied = state_to_cpu(original)

    assert copied.data_ptr() != original.data_ptr()
    original.fill_(5.0)
    assert torch.equal(copied, torch.ones(3))


def test_nested_structures_are_rebuilt_with_the_same_shape():
    state = {"a": [torch.zeros(1), (torch.ones(1), 3)], "b": {"c": "text", "d": None}}
    copied = state_to_cpu(state)

    assert isinstance(copied["a"], list)
    assert isinstance(copied["a"][1], tuple)
    assert copied["a"][1][1] == 3
    assert copied["b"] == {"c": "text", "d": None}


###############################################################################
################################### SNAPSHOT ##################################
###############################################################################


def test_snapshot_collects_one_entry_per_member(signature):
    trainers = [make_trainer(seed=0, epochs=2), make_trainer(seed=1, epochs=5)]
    checkpoint = make_checkpointer(signature, trainers).snapshot()

    assert isinstance(checkpoint, Checkpoint)
    assert checkpoint.signature is signature
    assert len(checkpoint.model_states) == 2
    assert checkpoint.training_state["epochs"] == [2, 5]
    assert len(checkpoint.training_state["optimizer_states"]) == 2
    assert [s["last_epoch"] for s in checkpoint.training_state["epoch_scheduler_states"]] == [2, 5]


def test_snapshot_is_independent_of_later_training(signature):
    trainer = make_trainer()
    checkpointer = make_checkpointer(signature, [trainer])
    checkpoint = checkpointer.snapshot()

    with torch.no_grad():
        trainer.model.weight.fill_(42.0)

    assert not torch.equal(checkpoint.model_states[0]["weight"], trainer.model.weight)


def test_component_signatures_are_captured_from_the_first_trainer(signature):
    trainers = [make_trainer(seed=0, lr=0.1), make_trainer(seed=1, lr=0.5)]
    checkpoint = make_checkpointer(signature, trainers).snapshot()

    assert checkpoint.training_state["optimizer_signature"] == {"trainer": {"optimizer_type": "adam", "lr": 0.1}}
    assert checkpoint.training_state["scheduler_signature"] == {
        "trainer": {"lr_scheduler": {"lr_scheduler_type": "step"}, "lr_warmup": False}
    }


@pytest.mark.parametrize("warmup", [False, True])
def test_warmup_state_is_only_recorded_when_a_warmup_scheduler_exists(signature, warmup):
    checkpoint = make_checkpointer(signature, [make_trainer(warmup=warmup)]).snapshot()

    warmup_state = checkpoint.training_state["warmup_scheduler_states"][0]
    assert (warmup_state is not None) is warmup
    assert checkpoint.training_state["scheduler_signature"]["trainer"]["lr_warmup"] is warmup


###############################################################################
############################## MEMBER BOOKKEEPING #############################
###############################################################################


def test_finished_members_are_served_from_their_cached_state(signature):
    first, second = make_trainer(seed=0, epochs=4), make_trainer(seed=1, epochs=1)
    trainers = [first, second]
    checkpointer = make_checkpointer(signature, trainers)

    expected_first = {k: v.clone() for k, v in first.model.state_dict().items()}
    checkpointer.finish_member(first)
    trainers[0] = None  # the strategy releases the finished trainer

    checkpoint = checkpointer.snapshot()

    assert checkpoint.training_state["epochs"] == [4, 1]
    for name, tensor in expected_first.items():
        assert torch.equal(checkpoint.model_states[0][name], tensor)


def test_cached_state_wins_over_a_still_present_trainer(signature):
    trainer = make_trainer(epochs=2)
    checkpointer = make_checkpointer(signature, [trainer])
    checkpointer.finish_member(trainer)
    trainer.completed_epochs = 99  # later mutation must not alter the cached snapshot

    assert checkpointer.snapshot().training_state["epochs"] == [2]


###############################################################################
################################### SAVING ####################################
###############################################################################


def test_save_returns_the_minimal_checkpoint_path_and_the_full_sibling_carries_the_snapshot(signature, tmp_path):
    path = make_checkpointer(signature, [make_trainer(epochs=3)]).save(tmp_path / "final.pth")

    assert path == tmp_path / "final.pth"
    assert path.is_file()
    assert Checkpoint.load(tmp_path / "final_full.pth").training_state["epochs"] == [3]


def test_save_writes_per_member_files_for_ensembles_by_default(signature, tmp_path):
    checkpointer = make_checkpointer(signature, [make_trainer(0), make_trainer(1)])
    checkpointer.save(tmp_path / "final.pth")

    assert (tmp_path / "final_member_0.pth").is_file()
    assert (tmp_path / "final_member_1_full.pth").is_file()


def test_save_separate_can_be_disabled(signature, tmp_path):
    make_checkpointer(signature, [make_trainer(0), make_trainer(1)]).save(tmp_path / "final.pth", save_separate=False)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["final.pth", "final_full.pth"]


###############################################################################
############################## PERIODIC CHECKPOINTS ###########################
###############################################################################


def test_step_is_disabled_without_directory_or_interval(signature, tmp_path):
    trainers = [make_trainer()]
    assert make_checkpointer(signature, trainers, save_dir=None, save_interval=1).step(0) is None
    assert make_checkpointer(signature, trainers, save_dir=tmp_path, save_interval=None).step(0, force_save=True) is None
    assert list(tmp_path.iterdir()) == []


def test_directory_is_only_created_when_periodic_saving_is_enabled(signature, tmp_path):
    make_checkpointer(signature, [], save_dir=tmp_path / "off", save_interval=None)
    make_checkpointer(signature, [], save_dir=tmp_path / "on" / "nested", save_interval=5)
    assert not (tmp_path / "off").exists()
    assert (tmp_path / "on" / "nested").is_dir()


@pytest.mark.parametrize(
    "epoch, saved",
    [(0, True), (1, False), (2, False), (3, True), (4, False), (6, True)],
)
def test_step_saves_on_multiples_of_the_interval(signature, tmp_path, epoch, saved):
    checkpointer = make_checkpointer(signature, [make_trainer()], save_dir=tmp_path, save_interval=3)
    name = checkpointer.step(epoch)
    assert (name is not None) is saved
    if saved:
        assert (tmp_path / name).is_file()


def test_force_save_overrides_the_interval(signature, tmp_path):
    checkpointer = make_checkpointer(signature, [make_trainer()], save_dir=tmp_path, save_interval=10)
    assert checkpointer.step(7) is None
    assert checkpointer.step(7, force_save=True) == "checkpoint_7_member_0.pth"
    assert (tmp_path / "checkpoint_7_member_0.pth").is_file()


def test_checkpoint_names_follow_the_active_member(signature, tmp_path):
    first, second = make_trainer(0), make_trainer(1)
    checkpointer = make_checkpointer(signature, [first, second], save_dir=tmp_path, save_interval=1)

    assert checkpointer.step(0) == "checkpoint_0_member_0.pth"
    checkpointer.finish_member(first)
    assert checkpointer.step(0) == "checkpoint_0_member_1.pth"
