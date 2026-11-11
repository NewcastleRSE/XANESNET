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

"""Unit tests for ``Checkpoint``: state-dict conversion, persistence and per-member splitting."""

import pytest
import torch

from xanesnet.serialization.checkpoints import Checkpoint
from xanesnet.serialization.config import Config


class _Unsafe:
    """Arbitrary object that ``weights_only`` unpickling must refuse to construct."""


def _model_state(value: float) -> dict[str, torch.Tensor]:
    return {"linear.weight": torch.full((2, 3), value), "linear.bias": torch.full((2,), value)}


def _training_state(n_members: int) -> dict:
    return {
        "epochs": list(range(10, 10 + n_members)),
        "optimizer_signature": {"trainer": {"optimizer_type": "adam"}},
        "optimizer_states": [{"step": i} for i in range(n_members)],
        "scheduler_signature": {"trainer": {"lr_warmup": False}},
        "epoch_scheduler_states": [{"last_epoch": i} for i in range(n_members)],
        "warmup_scheduler_states": [None for _ in range(n_members)],
    }


@pytest.fixture
def signature() -> Config:
    return Config({"model": {"model_type": "mlp", "in_size": 4}, "encodings": [{"encoding_type": "identity"}]})


@pytest.fixture
def full_checkpoint(signature) -> Checkpoint:
    return Checkpoint(signature, [_model_state(1.0)], _training_state(1))


@pytest.fixture
def ensemble_checkpoint(signature) -> Checkpoint:
    return Checkpoint(signature, [_model_state(float(i)) for i in range(3)], _training_state(3))


###############################################################################
############################## STATE-DICT CONVERSION ##########################
###############################################################################


def test_state_dict_contains_signature_as_plain_dict(signature):
    state = Checkpoint(signature, [_model_state(1.0)]).to_state_dict()
    assert state["signature"] == signature.as_dict()
    assert isinstance(state["signature"], dict)


def test_state_dict_omits_training_state_when_absent(signature):
    assert "training_state" not in Checkpoint(signature, [_model_state(1.0)]).to_state_dict()


def test_state_dict_round_trip_restores_equal_checkpoint(full_checkpoint):
    restored = Checkpoint.from_state_dict(full_checkpoint.to_state_dict())

    assert isinstance(restored.signature, Config)
    assert restored.signature.as_dict() == full_checkpoint.signature.as_dict()
    assert restored.training_state == full_checkpoint.training_state
    torch.testing.assert_close(restored.model_states[0]["linear.weight"], torch.full((2, 3), 1.0))


def test_from_state_dict_without_training_state_gives_none(signature):
    restored = Checkpoint.from_state_dict({"signature": signature.as_dict(), "model_states": []})
    assert restored.training_state is None


###############################################################################
################################# PERSISTENCE #################################
###############################################################################


def test_save_writes_minimal_and_full_files(full_checkpoint, tmp_path):
    path = full_checkpoint.save(tmp_path / "final.pth")

    assert path == tmp_path / "final.pth"
    assert path.is_file()
    assert (tmp_path / "final_full.pth").is_file()


def test_minimal_file_carries_no_training_state(full_checkpoint, tmp_path):
    full_checkpoint.save(tmp_path / "final.pth")
    assert Checkpoint.load(tmp_path / "final.pth").training_state is None


def test_full_file_restores_training_state(full_checkpoint, tmp_path):
    full_checkpoint.save(tmp_path / "final.pth")
    assert Checkpoint.load(tmp_path / "final_full.pth").training_state == full_checkpoint.training_state


def test_save_without_training_state_writes_only_the_minimal_file(signature, tmp_path):
    Checkpoint(signature, [_model_state(1.0)]).save(tmp_path / "final.pth")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["final.pth"]


def test_load_restores_weights_and_signature_exactly(full_checkpoint, tmp_path):
    full_checkpoint.save(tmp_path / "final.pth")

    loaded = Checkpoint.load(tmp_path / "final.pth")

    assert loaded.signature.as_dict() == full_checkpoint.signature.as_dict()
    assert len(loaded.model_states) == 1
    for name, tensor in full_checkpoint.model_states[0].items():
        torch.testing.assert_close(loaded.model_states[0][name], tensor)


def test_load_refuses_files_that_would_need_arbitrary_unpickling(tmp_path):
    """Checkpoints are loaded with ``weights_only=True`` so unsafe pickles must not execute."""

    path = tmp_path / "unsafe.pth"
    torch.save({"signature": {}, "model_states": [], "payload": _Unsafe()}, path)

    with pytest.raises(Exception, match="(?i)weights_only|unpickl"):
        Checkpoint.load(path)


###############################################################################
############################### SEPARATE MEMBERS ##############################
###############################################################################


def test_save_separate_is_a_noop_for_single_model_checkpoints(full_checkpoint, tmp_path):
    assert full_checkpoint.save_separate(tmp_path / "final.pth") == []
    assert list(tmp_path.iterdir()) == []


def test_save_separate_writes_one_pair_per_member(ensemble_checkpoint, tmp_path):
    paths = ensemble_checkpoint.save_separate(tmp_path / "final.pth")

    assert [p.name for p in paths] == [f"final_member_{i}.pth" for i in range(3)]
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [f"final_member_{i}.pth" for i in range(3)] + [f"final_member_{i}_full.pth" for i in range(3)]
    )


def test_each_member_checkpoint_holds_only_its_own_state(ensemble_checkpoint, tmp_path):
    paths = ensemble_checkpoint.save_separate(tmp_path / "final.pth")

    for index, path in enumerate(paths):
        member = Checkpoint.load(path)
        assert len(member.model_states) == 1
        torch.testing.assert_close(member.model_states[0]["linear.weight"], torch.full((2, 3), float(index)))
        assert member.signature.as_dict() == ensemble_checkpoint.signature.as_dict()


def test_member_training_state_is_sliced_per_member_but_shared_values_are_kept(ensemble_checkpoint, tmp_path):
    paths = ensemble_checkpoint.save_separate(tmp_path / "final.pth")

    member = Checkpoint.load(paths[1].with_name("final_member_1_full.pth"))
    assert member.training_state == {
        "epochs": [11],
        "optimizer_signature": {"trainer": {"optimizer_type": "adam"}},
        "optimizer_states": [{"step": 1}],
        "scheduler_signature": {"trainer": {"lr_warmup": False}},
        "epoch_scheduler_states": [{"last_epoch": 1}],
        "warmup_scheduler_states": [None],
    }


def test_save_separate_does_not_mutate_the_source_checkpoint(ensemble_checkpoint, tmp_path):
    before = {key: list(value) if isinstance(value, list) else value for key, value in ensemble_checkpoint.training_state.items()}
    ensemble_checkpoint.save_separate(tmp_path / "final.pth")
    assert ensemble_checkpoint.training_state == before
    assert len(ensemble_checkpoint.model_states) == 3


def test_save_separate_without_training_state_writes_minimal_files_only(signature, tmp_path):
    checkpoint = Checkpoint(signature, [_model_state(0.0), _model_state(1.0)])
    paths = checkpoint.save_separate(tmp_path / "final.pth")
    assert [p.name for p in paths] == ["final_member_0.pth", "final_member_1.pth"]
    assert len(list(tmp_path.iterdir())) == 2
